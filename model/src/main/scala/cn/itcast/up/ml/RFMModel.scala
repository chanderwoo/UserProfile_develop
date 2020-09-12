package cn.itcast.up.ml


import cn.itcast.up.base.BaseModel
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._

import scala.collection.immutable

/**
  * Author itcast
  * Date 2019/11/2 15:45
  * Desc
  */
object RFMModel extends BaseModel{
  def main(args: Array[String]): Unit = {
    execute()
  }
  /**
    * 获取标签id(即模型id,该方法应该在编写不同模型时进行实现)
    * @return
    */
  override def getTagID(): Int = 37

  /**
    * 开始计算
    * @param fiveDF  MySQL中的5级规则 id,rule
    * @param hbaseDF 根据selectFields查询出来的HBase中的数据
    * @return userid,tagIds
    */
  override def compute(fiveDF: DataFrame, hbaseDF: DataFrame): DataFrame = {
    //fiveDF.show(10)
    //fiveDF.printSchema()

    //hbaseDF.show(10)
    //hbaseDF.printSchema()
    /*
+---+----+
| id|rule|
+---+----+
| 38|   1|
| 39|   2|
| 40|   3|
| 41|   4|
| 42|   5|
| 43|   6|
| 44|   7|
+---+----+

root
 |-- id: long (nullable = false)
 |-- rule: string (nullable = true)

+---------+-------------------+-----------+----------+
| memberId|            orderSn|orderAmount|finishTime|
+---------+-------------------+-----------+----------+
| 13823431| ts_792756751164275|    2479.45|1564415022|
|  4035167| D14090106121770839|    2449.00|1565687310|
|  4035291| D14090112394810659|    1099.42|1564681801|
|  4035041| fx_787749561729045|    1999.00|1565799378|
| 13823285| D14092120154435903|    2488.00|1565062072|
|  4034219| D14092120155620305|    3449.00|1563601306|
|138230939|top_810791455519102|    1649.00|1565509622|
|  4035083| D14092120161884409|       7.00|1565731851|
|138230935| D14092120162313538|    1299.00|1565382991|
| 13823231| D14092120162378713|     499.00|1565677650|
+---------+-------------------+-----------+----------+
only showing top 10 rows

root
 |-- memberId: string (nullable = true)
 |-- orderSn: string (nullable = true)
 |-- orderAmount: string (nullable = true)
 |-- finishTime: string (nullable = true)
     */

    //0.定义常量字符串,避免后续拼写错误
    val recencyStr = "recency"
    val frequencyStr = "frequency"
    val monetaryStr = "monetary"
    val featureStr = "feature"
    val predictStr = "predict"


    //https://blog.csdn.net/liam08/article/details/79663018
    import org.apache.spark.sql.functions._
    import spark.implicits._

    //1.统计每个用户的RFM
    //Rencency:最近一次消费,最后一次订单距今时间
    //Frequency:消费频率,订单总数量
    //Monetary:消费金额,订单总金额

    //max("finishTime"))求每个用户的最近一次的订单时间
    //from_unixtime()将字符串时间戳转为时间对象
    //datediff(结束时间,开始时间)获取两个时间的时间差--天数
    val rencencyAggColumn: Column = functions.datediff(date_sub(current_timestamp(),60),from_unixtime(max("finishTime"))) as recencyStr
    val frequencyAggColumn: Column = functions.count("orderSn") as frequencyStr
    val monetaryAggColumn: Column = functions.sum("orderAmount") as monetaryStr

    val tempDF: DataFrame = hbaseDF
      .groupBy("memberId")
      .agg(rencencyAggColumn, frequencyAggColumn, monetaryAggColumn)
    //tempDF.show(10,false)

    /*
+---------+-------+---------+------------------+
|memberId |recency|frequency|monetary          |
+---------+-------+---------+------------------+
|13822725 |17     |116      |179298.34         |
|13823083 |17     |132      |233524.17         |
|138230919|17     |125      |240061.56999999998|
|13823681 |17     |108      |169746.1          |
|4033473  |17     |142      |251930.92         |
|13822841 |17     |113      |205931.91         |
|13823153 |17     |133      |250698.57         |
|13823431 |17     |122      |180858.22         |
|4033348  |17     |145      |240173.78999999998|
|4033483  |17     |110      |157811.09999999998|
+---------+-------+---------+------------------+
only showing top 10 rows
     */

    //2.数据打分(相当于归一化)
    //R: 1-3天=5分，4-6天=4分，7-9天=3分，10-15天=2分，大于16天=1分
    //F: ≥200=5分，150-199=4分，100-149=3分，50-99=2分，1-49=1分
    //M: ≥20w=5分，10-19w=4分，5-9w=3分，1-4w=2分，<1w=1分
    val recencyScore: Column = functions.when((col(recencyStr) >= 1) && (col(recencyStr) <= 3), 5)
      .when((col(recencyStr) >= 4) && (col(recencyStr) <= 6), 4)
      .when((col(recencyStr) >= 7) && (col(recencyStr) <= 9), 3)
      .when((col(recencyStr) >= 10) && (col(recencyStr) <= 15), 2)
      .when(col(recencyStr) >= 16, 1)
      .as(recencyStr)

    val frequencyScore: Column = functions.when(col(frequencyStr) >= 200, 5)
      .when((col(frequencyStr) >= 150) && (col(frequencyStr) <= 199), 4)
      .when((col(frequencyStr) >= 100) && (col(frequencyStr) <= 149), 3)
      .when((col(frequencyStr) >= 50) && (col(frequencyStr) <= 99), 2)
      .when((col(frequencyStr) >= 1) && (col(frequencyStr) <= 49), 1)
      .as(frequencyStr)

    val monetaryScore: Column = functions.when(col(monetaryStr) >= 200000, 5)
      .when(col(monetaryStr).between(100000, 199999), 4)
      .when(col(monetaryStr).between(50000, 99999), 3)
      .when(col(monetaryStr).between(10000, 49999), 2)
      .when(col(monetaryStr) <= 9999, 1)
      .as(monetaryStr)


    val RFMScoreDF: DataFrame = tempDF.select($"memberId",recencyScore,frequencyScore,monetaryScore)
    //RFMScoreDF.show(10,false)
    /*
+---------+-------+---------+--------+
|memberId |recency|frequency|monetary|
+---------+-------+---------+--------+
|13822725 |1      |3        |4       |
|13823083 |1      |3        |5       |
|138230919|1      |3        |5       |
|13823681 |1      |3        |4       |
|4033473  |1      |3        |5       |
|13822841 |1      |3        |5       |
|13823153 |1      |3        |5       |
|13823431 |1      |3        |4       |
|4033348  |1      |3        |5       |
|4033483  |1      |3        |4       |
+---------+-------+---------+--------+
     */
    //再使用SparkMLlib中的归一化工具进行数据缩放到[0,1]之间也可以


    //3.我们在使用机器学习算法的时候要的是向量,所以先对特征进行向量化表示
    //vectorAssembler可以将多列数据转换成一列特征的向量表示(特征向量)
    val vectorAssembler: VectorAssembler = new VectorAssembler()
      .setInputCols(Array(recencyStr, frequencyStr, monetaryStr))
      .setOutputCol(featureStr)
    val VectorDF: DataFrame = vectorAssembler.transform(RFMScoreDF)
    //VectorDF.show(10,false)
/*
+---------+-------+---------+--------+-------------+
|memberId |recency|frequency|monetary|feature      |
+---------+-------+---------+--------+-------------+
|13822725 |1      |3        |4       |[1.0,3.0,4.0]|
|13823083 |1      |3        |5       |[1.0,3.0,5.0]|
|138230919|1      |3        |5       |[1.0,3.0,5.0]|
|13823681 |1      |3        |4       |[1.0,3.0,4.0]|
|4033473  |1      |3        |5       |[1.0,3.0,5.0]|
|13822841 |1      |3        |5       |[1.0,3.0,5.0]|
|13823153 |1      |3        |5       |[1.0,3.0,5.0]|
|13823431 |1      |3        |4       |[1.0,3.0,4.0]|
|4033348  |1      |3        |5       |[1.0,3.0,5.0]|
|4033483  |1      |3        |4       |[1.0,3.0,4.0]|
+---------+-------+---------+--------+-------------+
 */

    //4.聚类
    //训练
    val model: KMeansModel = new KMeans()
      .setFeaturesCol(featureStr)
      .setPredictionCol(predictStr)
      .setK(7) //实际业务运营要求分为7类
      .setMaxIter(10) //最大迭代次数
      .setSeed(10) //可重复的随子种子
      .fit(VectorDF)
    //预测
    val result: DataFrame = model.transform(VectorDF)
    //result.show(10,false)
    /*
 +---------+-------+---------+--------+-------------+-------+
|memberId |recency|frequency|monetary|feature      |predict|
+---------+-------+---------+--------+-------------+-------+
|13822725 |1      |3        |4       |[1.0,3.0,4.0]|1      |
|13823083 |1      |3        |5       |[1.0,3.0,5.0]|0      |
|138230919|1      |3        |5       |[1.0,3.0,5.0]|0      |
|13823681 |1      |3        |4       |[1.0,3.0,4.0]|1      |
|4033473  |1      |3        |5       |[1.0,3.0,5.0]|0      |
|13822841 |1      |3        |5       |[1.0,3.0,5.0]|0      |
|13823153 |1      |3        |5       |[1.0,3.0,5.0]|0      |
|13823431 |1      |3        |4       |[1.0,3.0,4.0]|1      |
|4033348  |1      |3        |5       |[1.0,3.0,5.0]|0      |
|4033483  |1      |3        |4       |[1.0,3.0,4.0]|1      |
+---------+-------+---------+--------+-------------+-------+
only showing top 10 rows
     */

    //5.测试时看下聚类效果
    val ds: Dataset[Row] = result
      .groupBy(predictStr)
      .agg(max(col(recencyStr) + col(frequencyStr) + col(monetaryStr)), min(col(recencyStr) + col(frequencyStr) + col(monetaryStr)))
      .sort(col(predictStr).asc)
    //ds.show()
    /*
 +-------+---------------------------------------+---------------------------------------+
|predict|max(((recency + frequency) + monetary))|min(((recency + frequency) + monetary))|
+-------+---------------------------------------+---------------------------------------+
|      0|                                      9|                                      9|
|      1|                                      8|                                      8|
|      2|                                      3|                                      3|
|      3|                                      7|                                      7|
|      4|                                     11|                                     10|
|      5|                                      5|                                      4|
|      6|                                      8|                                      7|
+-------+---------------------------------------+---------------------------------------+
     */

    //6.获取聚类编号和聚类中心
    //接下来我们要对聚类之后的用户打Tag,不能仅仅根据聚类的序号打Tag
    //因为聚类序号并不能代表客户的价值等级,和我们5级规则并不对应
    //所以我们应该求出每个聚类的聚类中心,排序,聚类中心的值越大说明该用户群体价值越高

    //model.clusterCenters.indices获取所有聚类中心的索引编号
    //model.clusterCenters(i)根据索引取聚类中心
    //model.clusterCenters(i).toArray.sum求该聚类中心的RFM的和
    //[(聚类索引/编号, 聚类中心的RFM的和)]
    val indexAndRFM: immutable.IndexedSeq[(Int, Double)] = for(i <- model.clusterCenters.indices) yield (i,model.clusterCenters(i).toArray.sum)

    val tuples: immutable.IndexedSeq[(Int, Double)] = model.clusterCenters.indices.map(i=>(i,model.clusterCenters(i).toArray.sum))

    //indexAndRFM.foreach(println)
    /*
(0,9.0)
(1,8.0)
(2,3.0)
(3,7.0)
(4,10.038461538461538)
(5,4.333333333333333)
(6,7.5)

     */
    //println("==================")
    //根据聚类中心的RFM的和排好序的[(聚类索引/编号, 聚类中心的RFM的和)]
    val sortedIndexAndRFM: immutable.IndexedSeq[(Int, Double)] = indexAndRFM.sortBy(_._2).reverse
    //sortedIndexAndRFM.foreach(println)
/*
(4,10.038461538461538)
(0,9.0)
(1,8.0)
(6,7.5)
(3,7.0)
(5,4.333333333333333)
(2,3.0)
 */
    //7.将上面的排好序的聚类编号和聚类中心与5级规则进行对应
    /*
5级规则fiveDF:
+---+----+
| id|rule|
+---+----+
| 38|   1|
| 39|   2|
| 40|   3|
| 41|   4|
| 42|   5|
| 43|   6|
| 44|   7|
+---+----+
     */
    //目标:
    /*
+-------+---
predict| id|
+------+----
|    4| 38|
|    0| 39|
|    1| 40|
|    6| 41|
|    3| 42|
|    5| 43|
|    2| 44|
+--------+--
     */
    val indexAndRFMDS: Dataset[(Int, Double)] = sortedIndexAndRFM.toDS()
    val fiveDS: Dataset[(Long, String)] = fiveDF.as[(Long,String)]
    val tempRDD: RDD[((Int, Double), (Long, String))] = indexAndRFMDS.rdd.repartition(1).zip(fiveDS.rdd.repartition(1))
    //tempRDD.collect().foreach(println)
/*
((4,10.038461538461538),(38,1))
((0,9.0),(39,2))
((1,8.0),(40,3))
((6,7.5),(41,4))
((3,7.0),(42,5))
((5,4.333333333333333),(43,6))
((2,3.0),(44,7))
 */
    val ruleDF: DataFrame = tempRDD.map(t=>(t._1._1,t._2._1)).toDF("predict","tagIds")
    //ruleDF.show()
/*
+-------+------+
|predict|tagIds|
+-------+------+
|      4|    38|
|      0|    39|
|      1|    40|
|      6|    41|
|      3|    42|
|      5|    43|
|      2|    44|
+-------+------+
 */
    val ruleMap: collection.Map[Int, Long] = ruleDF.as[(Int,Long)].rdd.collectAsMap()

    val predict2Tag = udf((predict:Int)=>{
      ruleMap(predict)
    })

    val newDF: DataFrame = result.select($"memberId".as("userId"),predict2Tag('predict).as("tagIds"))
    newDF.show()
    /*
 +---------+------+
|   userId|tagIds|
+---------+------+
| 13822725|    40|
| 13823083|    39|
|138230919|    39|
| 13823681|    40|
|  4033473|    39|
| 13822841|    39|
| 13823153|    39|
| 13823431|    40|
|  4033348|    39|
|  4033483|    40|
|  4033575|    39|
|  4034191|    39|
|  4034923|    40|
| 13823077|    39|
|138230937|    40|
|  4034761|    39|
|  4035131|    40|
| 13822847|    39|
|138230911|    39|
|  4034221|    39|
+---------+------+
     */

    newDF
  }
}
