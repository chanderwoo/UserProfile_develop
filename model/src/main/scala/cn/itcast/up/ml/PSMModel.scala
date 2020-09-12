package cn.itcast.up.ml

import cn.itcast.up.base.BaseModel
import cn.itcast.up.common.HDFSUtils
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Column, DataFrame, Dataset, functions}

import scala.collection.immutable

/**
  * Author itcast
  * Date 2019/11/5 10:04
  * Desc 价格敏感度模型Price Sensitivity Meter
  * 有时在实际业务中，会把用户分为3-5类，
  * 比如分为极度敏感、较敏感、一般敏感、较不敏感、极度不敏感。
  * 然后将每类的聚类中心值与实际业务所需的其他指标结合，最终确定人群类别，判断在不同需求下是否触达或怎样触达。
  * 比如电商要通过满减优惠推广一新品牌的麦片，
  * 此时可优先选择优惠敏感且对麦片有消费偏好的用户进行精准推送，
  * 至于优惠敏感但日常对麦片无偏好的用户可暂时不进行推送或减小推送力度，
  * 优惠不敏感且对麦片无偏好的用户可选择不进行推送。
  * 可见，在实际操作中，技术指标评价外，还应结合业务需要，才能使模型达到理想效果。
  * //价格敏感度模型
  * //ra:receivableAmount 应收金额
  * //da:discountAmount 优惠金额
  * //pa:practicalAmount 实收金额
  *
  * //tdon 优惠订单数
  * //ton  总订单总数
  *
  * //ada 平均优惠金额
  * //ara 平均每单应收
  *
  * //tda 优惠总金额
  * //tra 应收总金额
  *
  * //tdonr 优惠订单占比(优惠订单数 / 订单总数)
  * //adar  平均优惠金额占比(平均优惠金额 / 平均每单应收金额)
  * //tdar  优惠总金额占比(优惠总金额 / 订单总金额)
  *
  * //psm = 优惠订单占比 + 平均优惠金额占比 + 优惠总金额占比
  * //psmScore = tdonr + adar + tdar
  * //简单来说可以使用占比和的值判断用户属于哪个类别,但是聚类效果完全取决于区间界限设置,所以还是用聚类算法
  * //>=1        极度敏感
  * //0.4~1      比较敏感
  * //0.1~0.3    一般敏感
  * //0          不太敏感
  * //<0         极度不敏感
  */
object PSMModel extends BaseModel{
  def main(args: Array[String]): Unit = {
    execute()
  }
  /**
    * 获取标签id(即模型id,该方法应该在编写不同模型时进行实现)
    * @return
    */
  override def getTagID(): Int = 50

  /**
    * 开始计算
    * @param fiveDF  MySQL中的5级规则 id,rule
    * @param hbaseDF 根据selectFields查询出来的HBase中的数据
    * @return userid,tagIds
    */
  override def compute(fiveDF: DataFrame, hbaseDF: DataFrame): DataFrame = {
    //fiveDF.show(10,false)
    //fiveDF.printSchema()

    //hbaseDF.show(100,false)
    //hbaseDF.printSchema()

    /*
+---+----+
|id |rule|
+---+----+
|51 |1   |
|52 |2   |
|53 |3   |
|54 |4   |
|55 |5   |
+---+----+

root
 |-- id: long (nullable = false)
 |-- rule: string (nullable = true)

+---------+-------------------+-----------+---------------+
|memberId |orderSn            |orderAmount|couponCodeValue|
+---------+-------------------+-----------+---------------+
|13823431 |ts_792756751164275 |2479.45    |0.00           |
|4035167  |D14090106121770839 |2449.00    |0.00           |
|4035291  |D14090112394810659 |1099.42    |0.00           |
|4035041  |fx_787749561729045 |1999.00    |0.00           |
|13823285 |D14092120154435903 |2488.00    |0.00           |
|4034219  |D14092120155620305 |3449.00    |0.00           |
|138230939|top_810791455519102|1649.00    |0.00           |
|4035083  |D14092120161884409 |7.00       |0.00           |
|138230935|D14092120162313538 |1299.00    |0.00           |
|13823231 |D14092120162378713 |499.00     |0.00           |
+---------+-------------------+-----------+---------------+
only showing top 10 rows

root
 |-- memberId: string (nullable = true)
 |-- orderSn: string (nullable = true)
 |-- orderAmount: string (nullable = true)
 |-- couponCodeValue: string (nullable = true)
     */

    import org.apache.spark.sql.functions._
    import spark.implicits._


    //0.定义常量
    // psmScore = tdonr + adar + tdar
    val psmScoreStr: String = "psm"
    val featureStr: String = "feature"
    val predictStr: String = "predict"

    //1.计算指标
    //ra:receivableAmount 应收金额
    val raColumn:Column = ('orderAmount + 'couponCodeValue) as "ra"
    //da:discountAmount 优惠金额
    val daColumn:Column = 'couponCodeValue as "da"
    //pa:practicalAmount 实收金额
    val paColumn:Column = 'orderAmount as "pa"
    //订单状态,1表示优惠订单即couponCodeValue != 0
    val state:Column = functions
      .when('couponCodeValue =!= 0.0D, 1)//不等于
      .when('couponCodeValue === 0.0d, 0)
      .as("state")

    //tdon 优惠订单数
    val tdon:Column = sum('state) as "tdon"
    //ton  总订单总数
    val ton:Column = count('state) as "ton"

    //tda 优惠总金额
    val tda: Column  = sum('da) as "tda"
    //tra 应收总金额
    val tra: Column = sum('ra) as "tra"

    //ada 平均优惠金额 === ('tda / 'tdon) //tdon优惠订单数可能为0
    //ara 平均每单应收 === ('tra / 'ton)

    val tempDF: DataFrame = hbaseDF.select('memberId, raColumn, daColumn, paColumn, state)
      .groupBy('memberId)
      .agg(tdon, ton, tda, tra)
    //tempDF.show(10,false)
    /*
 +---------+----------+------------+------+------------------+
|memberId |tdon       |ton        |tda   |tra               |
+---------+----------+------------+------+------------------+
|4033473  |3         |142         |500.0 |252430.92         |
|13822725 |4         |116         |800.0 |180098.34         |
|13823681 |1         |108         |200.0 |169946.1          |
|138230919|3         |125         |600.0 |240661.56999999998|
|13823083 |3         |132         |600.0 |234124.17         |
|13823431 |2         |122         |400.0 |181258.22         |
|4034923  |1         |108         |200.0 |167674.89         |
|4033575  |4         |125         |650.0 |255866.40000000002|
|13822841 |0         |113         |0.0   |205931.91         |
|13823153 |6         |133         |1200.0|251898.57         |
+---------+----------+------------+------+------------------+
only showing top 10 rows
     */
    //3.计算psm
    //tdonr 优惠订单占比(优惠订单数 / 订单总数)
    //adar  平均优惠金额占比(平均优惠金额 / 平均每单应收金额)
    //tdar  优惠总金额占比(优惠总金额 / 订单总金额)
    //psm = 优惠订单占比 + 平均优惠金额占比 + 优惠总金额占比
    //psm的计算有很多除法,除数有可能为0,SparkSQL对于除数为0的记录直接返回null
    //注意:SparkMLlib在计算的时候不能有null记录,所以应该将null记录过滤掉
    //注意:SparkSQL的DSL语法中对于Null值的判断得使用isNotNull方法
    //用"null" null 都不行
    //注意:对于tdon优惠订单数为0,我们这里演示的知识点是机器学习需要处理null值,及如何处理
    //而对于tdon优惠订单数为0的用户实际上是对价格不敏感的用户,应该要保留
    //那么就可以将tdon优惠订单数为0的用户的用户的tdon优惠订单数置为一个很小的值,如0.00001
    val psmScoreColumn:Column = ('tdon / 'ton) + (('tda / 'tdon)/(('tra / 'ton))) + ('tda/'tra) as "psm"
    val psmScoreDF: DataFrame = tempDF.select('memberId,psmScoreColumn).filter('psm.isNotNull)
    psmScoreDF.show(10,false)
    /*
+---------+-------------------+
|memberId |psm                |
+---------+-------------------+
|4033473  |0.11686252330855691|
|13822725 |0.16774328728519597|
|13823681 |0.13753522440350205|
|138230919|0.1303734438365045 |
|13823083 |0.1380506927739941 |
|13823431 |0.15321482374431458|
|4034923  |0.13927276336831218|
|4033575  |0.11392752155030905|
|13823153 |0.15547466292943982|
|4034191  |0.11026694172505715|
+---------+-------------------+
only showing top 10 rows
     */

    //4.特征向量化
    val vectorDF: DataFrame = new VectorAssembler()
      .setInputCols(Array(psmScoreStr))
      .setOutputCol(featureStr).transform(psmScoreDF)

    var model: KMeansModel = null
    val path = "/model/PSMModel2/"
    //5.聚类
    if (HDFSUtils.getInstance().exists(path)){
      //model保存过,直接加载
      model= KMeansModel.load(path)
    }else{
      //model不存在,先训练再保存
      model = new KMeans()
        .setK(5)
        .setMaxIter(10)
        .setSeed(10)
        .setFeaturesCol(featureStr)
        .setPredictionCol(predictStr)
        .fit(vectorDF)//训练
      model.save(path)
    }

    //6.预测
    val result: DataFrame = model.transform(vectorDF)
    result.show(10,false)

    //7.获取聚类编号和聚类中心
    //接下来我们要对聚类之后的用户打Tag,不能仅仅根据聚类的序号打Tag
    //因为聚类序号并不能代表客户的价值等级,和我们5级规则并不对应
    //所以我们应该求出每个聚类的聚类中心,排序,聚类中心的值越大说明该用户群体价值越高

    //model.clusterCenters.indices获取所有聚类中心的索引编号
    //model.clusterCenters(i)根据索引取聚类中心
    //model.clusterCenters(i).toArray.sum求该聚类中心的RFM的和
    //[(聚类索引/编号, 聚类中心的RFM的和)]
    val indexAndRFM: immutable.IndexedSeq[(Int, Double)] = for(i <- model.clusterCenters.indices) yield (i,model.clusterCenters(i).toArray.sum)
    //val tuples: immutable.IndexedSeq[(Int, Double)] = model.clusterCenters.indices.map(i=>(i,model.clusterCenters(i).toArray.sum))
    indexAndRFM.foreach(println)
    /*
(0,12.0)
(1,13.0)
(2,14.0)
(3,13.0)
     */
    //println("==================")
    //根据聚类中心的RFM的和排好序的[(聚类索引/编号, 聚类中心的RFM的和)]
    val sortedIndexAndRFM: immutable.IndexedSeq[(Int, Double)] = indexAndRFM.sortBy(_._2).reverse
    //sortedIndexAndRFM.foreach(println)

    //8.将上面的排好序的聚类编号和聚类中心与5级规则进行对应

    val indexAndRFMDS: Dataset[(Int, Double)] = sortedIndexAndRFM.toDS()
    val fiveDS: Dataset[(Long, String)] = fiveDF.as[(Long,String)]
    val tempRDD: RDD[((Int, Double), (Long, String))] = indexAndRFMDS.rdd.repartition(1).zip(fiveDS.rdd.repartition(1))
    tempRDD.collect().foreach(println)
    /*
((2,14.0),(46,1))
((3,13.0),(47,2))
((1,13.0),(48,3))
((0,12.0),(49,4))
     */
    val ruleDF: DataFrame = tempRDD.map(t=>(t._1._1,t._2._1)).toDF("predict","tagIds")
    ruleDF.show()
    /*
+-------+------+
|predict|tagIds|
+-------+------+
|      2|    46|
|      3|    47|
|      1|    48|
|      0|    49|
+-------+------+
     */
    val ruleMap: collection.Map[Int, Long] = ruleDF.as[(Int,Long)].rdd.collectAsMap()

    val predict2Tag = udf((predict:Int)=>{
      ruleMap(predict)
    })

    val newDF: DataFrame = result.select($"memberId".as("userId"),predict2Tag('predict).as("tagIds"))
    newDF.show()
    /*

     */

    newDF
  }
}
