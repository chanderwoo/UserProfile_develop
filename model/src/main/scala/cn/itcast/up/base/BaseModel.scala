package cn.itcast.up.base

import java.util.Properties

import cn.itcast.up.bean.HBaseMeta
import com.typesafe.config.{Config, ConfigFactory}
import org.apache.commons.lang3.StringUtils
import org.apache.spark
import org.apache.spark.sql.{DataFrame, SparkSession}


/**
  * 抽象出模型计算的步骤,后续编写模型的时候只需要基础该类并实现抽象方法即可
  */
trait BaseModel {
  //0.封装参数
  val config: Config = ConfigFactory.load()
  val url: String = config.getString("jdbc.url")
  val tableName: String = config.getString("jdbc.table")
  val sourceClass: String = config.getString("hbase.source.class")
  val zkHosts: String = config.getString("hbase.source.zkHosts")
  val zkPort: String = config.getString("hbase.source.zkPort")
  val hbaseTable: String = config.getString("hbase.source.hbaseTable")
  val family: String = config.getString("hbase.source.family")
  val selectFields: String = config.getString("hbase.source.selectFields")
  val rowKey: String = config.getString("hbase.source.rowKey")

  val hbaseMeta = HBaseMeta(
    "",
    zkHosts,
    zkPort,
    hbaseTable,
    family,
    selectFields,
    rowKey
  )

  //0.创建sparksession
  val spark = SparkSession.builder()
    .appName("model")
    .master("local[*]")
    .config("spark.hadoop.validateOutputSpecs", "false")
    //.config("spark.local.dir","hdfs://bd001:8020/temp")
    //.config("spark.driver.memory","5g")
    .getOrCreate()
  spark.sparkContext.setLogLevel("WARN")

  //System.setProperty("HADOOP_USER_NAME","root")

  import spark.implicits._
  import scala.collection.JavaConversions._
  import org.apache.spark.sql.functions._

  /**
    * 封装执行流程,继承该类后,在main方法中调用该方法即可
    */
  def execute(): Unit ={
    //1.加载MySQL数据
    val mysqlDF: DataFrame = getMySQLSource()
    //2.获取4级规则
    val params: Map[String, String] = getFourDF(mysqlDF)
    //3.获取5级规则
    val fiveDF: DataFrame = getFiveDF(mysqlDF)
    //4.加载HBase数据
    val hbaseDF: DataFrame = getHBaseSource(params)
    //5.开始计算标签
    val newDF: DataFrame = compute(fiveDF, hbaseDF)
    //6.合并结果
    val resultDF: DataFrame = beginMergeTag(newDF)
    //7.保存结果
    saveData(resultDF)
  }



  /**
    * 加载MySQL数据
    * @return
    */
  def getMySQLSource(): DataFrame={
    val properties = new Properties()
    val mysqlDF: DataFrame = spark.read.jdbc(url, tableName, properties)
    mysqlDF
  }


  /**
    * 获取标签id(即模型id,该方法应该在编写不同模型时进行实现)
    * @return
    */
  def getTagID():Int


  /**
    * 获取4级数据源规则
    * @param mysqlDF
    * @return
    */
  def getFourDF(mysqlDF: DataFrame):Map[String, String]={
    mysqlDF.select('rule).where('id === getTagID())
      .rdd.map(row => {
        val ruleArr: Array[String] = row.getAs[String]("rule").split("##")
        ruleArr.map(kvStr => {
          val kv: Array[String] = kvStr.split("=")
          (kv(0), kv(1))
        }).toMap
    }).collect()(0)
  }

  /**
    * 获取5级数据源规则
    * @param mysqlDF
    * @return
    */
  def getFiveDF(mysqlDF: DataFrame): DataFrame={
    mysqlDF.select('id, 'rule).where('pid === getTagID()).toDF()
  }

  /**
    * 加载HBase数据
    * @param params
    * @return
    */
  def getHBaseSource(params: Map[String, String]): DataFrame={
    spark.read.format(sourceClass).options(params).load()
  }

  /**
    * 开始计算
    * @param fiveDF MySQL中的5级规则 id,rule
    * @param hbaseDF 根据selectFields查询出来的HBase中的数据
    * @return userid,tagIds
    */
  def compute(fiveDF: DataFrame, hbaseDF: DataFrame): DataFrame


  /**
    * 计算结果合并
    * @param newDF
    * @return
    */
  def beginMergeTag(newDF: DataFrame): DataFrame={
    //读取之前的数据
    val oldDF: DataFrame = spark.read
      .format(sourceClass)
      .option(HBaseMeta.ZKHOSTS, hbaseMeta.zkHosts)
      .option(HBaseMeta.ZKPORT, hbaseMeta.zkPort)
      .option(HBaseMeta.HBASETABLE, hbaseMeta.hbaseTable)
      .option(HBaseMeta.FAMILY, hbaseMeta.family)
      .option(HBaseMeta.SELECTFIELDS, hbaseMeta.selectFields)
      .option(HBaseMeta.ROWKEY, hbaseMeta.rowKey)
      .load()
    //将本次的数据合并
    //    2 34   2 43   2 34,43
    oldDF.createOrReplaceTempView("oldTbl")
    newDF.createOrReplaceTempView("newTbl")

    spark.udf.register("mergeTag",(oldTag: String, newTag: String)=>{
      if (StringUtils.isBlank(oldTag)){
        newTag
      } else if (StringUtils.isBlank(newTag)) {
        oldTag
      } else {
        //两个都不为空,开始合并
        (oldTag.split(",") ++ newTag.split(",")).toSet.mkString(",")
      }
    })

    val sql =
      """
        |select n.userId as userId, mergeTag(o.tagIds, n.tagIds) as tagIds
        |from newTbl n
        |left join oldTbl o
        |on o.userId = n.userId
      """.stripMargin

    val resultDF: DataFrame = spark.sql(sql)
    resultDF
  }

  /**
    * 数据落地到HBase
    * 将合并后的数据存入HBase
    * @param resultDF
    */
  def saveData(resultDF: DataFrame) = {
    resultDF.show(10)
    resultDF
      .write
      .format(sourceClass)
      .option(HBaseMeta.ZKHOSTS, hbaseMeta.zkHosts)
      .option(HBaseMeta.ZKPORT, hbaseMeta.zkPort)
      .option(HBaseMeta.HBASETABLE, hbaseMeta.hbaseTable)
      .option(HBaseMeta.FAMILY, hbaseMeta.family)
      .option(HBaseMeta.SELECTFIELDS, hbaseMeta.selectFields)
      .option(HBaseMeta.ROWKEY, hbaseMeta.rowKey)
      .save()
  }
}
