package cn.itcast.up.ml

import cn.itcast.up.base.BaseModel
import cn.itcast.up.common.HDFSUtils
import org.apache.hadoop.fs.Hdfs
import org.apache.hadoop.hdfs.client.HdfsUtils
import org.apache.solr.util.HdfsUtil
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Column, DataFrame, Dataset, TypedColumn}

import scala.collection.{immutable, mutable}

/**
  * Author itcast
  * Date 2019/11/3 15:01
  * Desc 用户活跃度模型-RFE
  * Recency:最近一次访问时间,用户最后一次访问距今时间
  * Frequency:访问频率,用户一段时间内访问的页面总次数,
  * Engagements:页面互动度,用户一段时间内访问的独立页面数,也可以定义为页面 浏览量、下载量、 视频播放数量等
  */
object FREModel extends BaseModel{

  def main(args: Array[String]): Unit = {
    execute()
  }
  /**
    * 获取标签id(即模型id,该方法应该在编写不同模型时进行实现)
    * @return
    */
  override def getTagID(): Int = 45

  /**
    * 开始计算
    * @param fiveDF  MySQL中的5级规则 id,rule
    * @param hbaseDF 根据selectFields查询出来的HBase中的数据
    * @return userid,tagIds
    */
  override def compute(fiveDF: DataFrame, hbaseDF: DataFrame): DataFrame = {
    //fiveDF.show(10,false)
    //fiveDF.printSchema()
    //hbaseDF.show(10,false)
    //hbaseDF.printSchema()
    /*
 +---+----+
|id |rule|
+---+----+
|46 |1   |
|47 |2   |
|48 |3   |
|49 |4   |
+---+----+

root
 |-- id: long (nullable = false)
 |-- rule: string (nullable = true)

+--------------+-------------------------------------------------------------------+-------------------+
|global_user_id|loc_url                                                            |log_time           |
+--------------+-------------------------------------------------------------------+-------------------+
|424           |http://m.eshop.com/mobile/coupon/getCoupons.html?couponsId=3377    |2019-08-13 03:03:55|
|619           |http://m.eshop.com/?source=mobile                                  |2019-07-29 15:07:41|
|898           |http://m.eshop.com/mobile/item/11941.html                          |2019-08-14 09:23:44|
|642           |http://www.eshop.com/l/2729-2931.html                              |2019-08-11 03:20:17|
|130           |http://www.eshop.com/                                              |2019-08-12 11:59:28|
|515           |http://www.eshop.com/l/2723-0-0-1-0-0-0-0-0-0-0-0.html             |2019-07-23 14:39:25|
|274           |http://www.eshop.com/                                              |2019-07-24 15:37:12|
|772           |http://ck.eshop.com/login.html                                     |2019-07-24 07:56:49|
|189           |http://m.eshop.com/mobile/item/9673.html                           |2019-07-26 19:17:00|
|529           |http://m.eshop.com/mobile/search/_bplvbiwq_XQS75_btX_ZY1328-se.html|2019-07-25 23:18:37|
+--------------+-------------------------------------------------------------------+-------------------+
only showing top 10 rows

root
 |-- global_user_id: string (nullable = true)
 |-- loc_url: string (nullable = true)
 |-- log_time: string (nullable = true)
     */

    //https://blog.csdn.net/liam08/article/details/79663018
    import org.apache.spark.sql.functions._
    import spark.implicits._

    //0.定义常量字符串,避免后续拼写错误
    val recencyStr = "recency"
    val frequencyStr = "frequency"
    val engagementsStr = "engagements"
    val featureStr = "feature"
    val scaleFeatureStr = "scaleFeature"
    val predictStr = "predict"

    //Recency:最近一次访问时间,用户最后一次访问距今时间
    //Frequency:访问频率,用户一段时间内访问的页面总次数,
    //Engagements:页面互动度,用户一段时间内访问的独立页面数,也可以定义为页面 浏览量、下载量、 视频播放数量等
    val recencyAggColumn: Column = datediff(date_sub(current_timestamp(),60),max("log_time")) as recencyStr
    val frequencyAggColumn: Column = count("loc_url") as frequencyStr
    val engagementsAggColumn: Column = countDistinct("loc_url") as engagementsStr

    //1.根据用户id进行分组,求出RFE
    val tempDF: DataFrame = hbaseDF.groupBy("global_user_id")
      .agg(recencyAggColumn, frequencyAggColumn, engagementsAggColumn)
    //tempDF.show(10,false)
    /*
+--------------+-------+---------+-----------+
|global_user_id|recency|frequency|engagements|
+--------------+-------+---------+-----------+
|296           |18     |380      |227        |
|467           |18     |405      |267        |
|675           |18     |370      |240        |
|691           |18     |387      |244        |
|829           |18     |404      |269        |
|125           |18     |375      |246        |
|451           |18     |347      |224        |
|800           |18     |395      |242        |
|853           |18     |388      |252        |
|944           |18     |394      |252        |
+--------------+-------+---------+-----------+
     */

    //2.打分
    // R:0-15天=5分，16-30天=4分，31-45天=3分，46-60天=2分，大于61天=1分
    // F:≥400=5分，300-399=4分，200-299=3分，100-199=2分，≤99=1分
    // E:≥250=5分，230-249=4分，210-229=3分，200-209=2分，1=1分
    val recencyScore: Column = when(col(recencyStr).between(0, 15), 5)
      .when(col(recencyStr).between(16, 30), 4)
      .when(col(recencyStr).between(31, 45), 3)
      .when(col(recencyStr).between(46, 60), 2)
      .when(col(recencyStr).gt(60), 1)
      .as(recencyStr)

    val frequencyScore: Column = when(col(frequencyStr).geq(400), 5)
      .when(col(frequencyStr).between(300, 399), 4)
      .when(col(frequencyStr).between(200, 299), 3)
      .when(col(frequencyStr).between(100, 199), 2)
      .when(col(frequencyStr).leq(99), 1)
      .as(frequencyStr)

    val engagementsScore: Column = when(col(engagementsStr).geq(250), 5)
      .when(col(engagementsStr).between(200, 249), 4)
      .when(col(engagementsStr).between(150, 199), 3)
      .when(col(engagementsStr).between(50, 149), 2)
      .when(col(engagementsStr).leq(49), 1)
      .as(engagementsStr)

    val FREScoreDF: DataFrame = tempDF.select($"global_user_id".as("userId"),recencyScore,frequencyScore,engagementsScore)
      .where('userId.isNotNull and col(recencyStr).isNotNull and col(frequencyStr).isNotNull and col(engagementsStr).isNotNull)
    //FREScoreDF.show(10,false)
/*
only showing top 10 rows
+------+-------+---------+-----------+
|userId|recency|frequency|engagements|
+------+-------+---------+-----------+
|296   |4      |4        |4          |
|467   |4      |5        |5          |
|675   |4      |4        |4          |
|691   |4      |4        |4          |
|829   |4      |5        |5          |
|125   |4      |4        |4          |
|451   |4      |4        |4          |
|800   |4      |4        |4          |
|853   |4      |4        |5          |
|944   |4      |4        |5          |
+------+-------+---------+-----------+
only showing top 10 rows
 */
    //再使用SparkMLlib中的归一化工具进行数据缩放到[0,1]之间也可以

    //3.特征向量化
    val vecotrDF: DataFrame = new VectorAssembler()
      .setInputCols(Array(recencyStr, frequencyStr, engagementsStr))
      .setOutputCol(featureStr)
      .transform(FREScoreDF)


    //4.聚类

    //4.1准备待使用的k
    val ks: List[Int] = List(2,3,4,5,6,7,8)
    //4.2准备一个集合存放k对应的SSE
    //集合内误差平方和:Within Set Sum of Squared Error, WSSSE/SSE
    val K2SSE: mutable.Map[Int, Double] = mutable.Map[Int,Double]()
    //4.3依次计算每个k对应的SSE
    for(k <- ks){
      val model: KMeansModel = new KMeans()
        .setK(k)
        .setMaxIter(10)
        .setSeed(10)
        .setFeaturesCol(featureStr)
        .setPredictionCol(predictStr)
        .fit(vecotrDF)
      val SSE: Double = model.computeCost(vecotrDF)
      K2SSE.put(k,SSE)
    }
    //4.4输出k的对应的SSE的值,根据肘部法则+运营/产品的业务要求确定最终的K值
    K2SSE.foreach(println)

    //5.模型的保存和加载
    //使用最优K进行模型训练,训练完之后可以对模型进行保存,方便下次直接使用
    var model: KMeansModel = null
    val path:String  = "/model/RFEModel2"
    if (HDFSUtils.getInstance().exists(path)){
      //如果模型目录存在应该直接加载使用
      println("模型目录存在直接加载")
      model= KMeansModel.load(path)
    }else{
      //如果模型目录不存在应该进行训练并保存
      println("模型目录不存在,开始进行训练")
      model = new KMeans()
        .setK(4)
        .setMaxIter(10)
        .setSeed(10)
        .setFeaturesCol(featureStr)
        .setPredictionCol(predictStr)
        .fit(vecotrDF)
      model.save(path)
    }

    //6.预测
    val result: DataFrame = model.transform(vecotrDF)
    result.show(10,false)
    /*
+------+-------+---------+-----------+-------------+-------+
|userId|recency|frequency|engagements|feature      |predict|
+------+-------+---------+-----------+-------------+-------+
|296   |4      |4        |4          |[4.0,4.0,4.0]|0      |
|467   |4      |5        |5          |[4.0,5.0,5.0]|2      |
|675   |4      |4        |4          |[4.0,4.0,4.0]|0      |
|691   |4      |4        |4          |[4.0,4.0,4.0]|0      |
|829   |4      |5        |5          |[4.0,5.0,5.0]|2      |
|125   |4      |4        |4          |[4.0,4.0,4.0]|0      |
|451   |4      |4        |4          |[4.0,4.0,4.0]|0      |
|800   |4      |4        |4          |[4.0,4.0,4.0]|0      |
|853   |4      |4        |5          |[4.0,4.0,5.0]|1      |
|944   |4      |4        |5          |[4.0,4.0,5.0]|1      |
+------+-------+---------+-----------+-------------+-------+
only showing top 10 rows
     */


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

    //7.将上面的排好序的聚类编号和聚类中心与5级规则进行对应

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

    val newDF: DataFrame = result.select($"userId",predict2Tag('predict).as("tagIds"))
    newDF.show()
    /*

     */

    newDF
  }
}
