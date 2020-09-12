package cn.itcast.up.base

import java.util.Properties

import cn.itcast.up.bean.HBaseMeta
import com.typesafe.config.{Config, ConfigFactory}
import org.apache.commons.lang3.StringUtils
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * 抽象出模型计算的步骤,后续编写模型的时候只需要基础该类并实现抽象方法即可
  */
trait BaseModel2 {
  //0.封装参数
  private val config: Config = ConfigFactory.load()
  private val mysqlURL: String = config.getString("jdbc.mysql.url")
  private val mysqlTableName: String = config.getString("jdbc.mysql.tablename")
  private val hbaseSource: String = config.getString("hbase.source")
  private val hbase_zkhosts: String = config.getString("hbase.zkhosts")
  private val hbase_zkport: String = config.getString("hbase.zkport")
  private val hbase_hbasetable: String = config.getString("hbase.hbasetable")
  private val hbase_family: String = config.getString("hbase.family")
  private val hbase_selectfields: String = config.getString("hbase.selectfields")
  private val hbase_rowkey: String = config.getString("hbase.rowkey")

  ////0.创建sparksession
  val spark: SparkSession = SparkSession.builder()
    .appName(setAppName())
    .master("local[*]")
    .getOrCreate()

  import spark.implicits._
  import scala.collection.JavaConversions._
  import org.apache.spark.sql.functions._

  /**
    * 封装执行流程,继承该类后,在main方法中调用该方法即可
    */
  def execute()={
    //1.加载MySQL数据源
    val mysqlDF: DataFrame = getMySQLDF()
    //2.获取4级标签规则
    val params: Map[String, String] = getFourRuleMap(mysqlDF)
    //3.获取5级标签规则
    val fiveDF: DataFrame = getFiveRuleDF(mysqlDF)
    //4.获取HBase数据源.
    val hbaseDF: DataFrame = getHBaseSourceDF(params)
    //开始进行标签计算
    val resultDF: DataFrame = computeTag(fiveDF, hbaseDF)
    //将计算结果保存.
    saveResult(resultDF)
  }






  /**
    * 定义当前应用的名称.
    * @return
    */
  def setAppName(): String



  /**
    * 计算(儿子实现)
    */
  def computeTag(fiveRuleDF: DataFrame, sourceDF: DataFrame): DataFrame


  /**
    * 加载MySQL数据
    * @return
    */
  def getMySQLDF(): DataFrame={
    val mysqlDF: DataFrame = spark.read.jdbc(mysqlURL, mysqlTableName, new Properties())
    mysqlDF
  }

  /**
    * 获取标签id(即模型id,该方法应该在编写不同模型时进行实现)
    * @return
    */
  def setTagID(): Int

  //加载4级标签规则
  def getFourRuleMap(mysqlDF: DataFrame): Map[String, String] = {
    val params: Map[String, String] =  mysqlDF.select('id, 'rule)
      .where('id === setTagID()).rdd
      .map(row => {
      val ruleSource: String = row.getAs("rule").toString
      ruleSource.split("##")
        .map(str => {
          val arr: Array[String] = str.split("=")
          (arr(0), arr(1))
        }).toMap
    }).collect()(0)
    params
  }

  /** 解析4级标签的元数据.
    * 将map转换为HBaseMeta
    * @param params
    * @return
    */
  def parseHBaseMeta(params: Map[String, String]): HBaseMeta = {
    HBaseMeta(
      params.getOrElse(HBaseMeta.INTYPE, ""),
      params.getOrElse(HBaseMeta.ZKHOSTS, ""),
      params.getOrElse(HBaseMeta.ZKPORT, ""),
      params.getOrElse(HBaseMeta.HBASETABLE, ""),
      params.getOrElse(HBaseMeta.FAMILY, ""),
      params.getOrElse(HBaseMeta.SELECTFIELDS, ""),
      params.getOrElse(HBaseMeta.ROWKEY, "")
    )
  }


  //加载5级标签规则(待定),我们不直接将5级转换为TagRule实体类,map
  def getFiveRuleDF(mysqlDF: DataFrame): DataFrame = {
    mysqlDF.select('id, 'rule)
      .where('pid === setTagID())
      .toDF()
  }


  //加载HBase数据源
  def getHBaseSourceDF(params: Map[String, String]): DataFrame = {
    spark.read
      .format(hbaseSource)
      .options(params)
      .load()
  }

  //数据落地.
  def saveResult(resultDF: DataFrame)={
    //先判断数据集是否为空
    if (resultDF == null) {
       null
    } else {
      //获取历史数据
      val oldDF: DataFrame = spark.read
        .format(hbaseSource)
        .option(HBaseMeta.ZKHOSTS, hbase_zkhosts)
        .option(HBaseMeta.ZKPORT, hbase_zkport)
        .option(HBaseMeta.HBASETABLE, hbase_hbasetable)
        .option(HBaseMeta.FAMILY, hbase_family)
        .option(HBaseMeta.SELECTFIELDS, hbase_selectfields)
        .load()
      //合并新老数据
      //自定义函数,进行标签的合并
      val mergeTag = udf((newTag: String, oldTag: String) => {
        //对新老数据进行非空判断
        if (StringUtils.isBlank(newTag) && StringUtils.isBlank(oldTag)){
          //如果两个都为空,返回""
          ""
        } else if (StringUtils.isBlank(newTag)) {
          oldTag
        } else if (StringUtils.isBlank(oldTag)) {
          newTag
        } else {
          //如果新老数据都不为空,那么就进行字符串拼接
          //222,444,555,666,222
          //性别4级标签:230 男:233 女:234
          // new:230-234 old:230-233
          val tagStr = newTag + "," + oldTag
          tagStr.split(",").toList
            //对标签进行去重.
            .toSet
            //将集合按照指定的分隔符进行分隔,然后合并为一个字符串
            .mkString(",")
        }

      })
      val fullDF: DataFrame = resultDF.join(oldDF,resultDF.col("userid") === oldDF.col("userid"),"full")
      val saveDF: DataFrame = fullDF.select(
        //userid,tagIds
        when(resultDF.col("userid").isNotNull, resultDF.col("userid"))
          .when(resultDF.col("userid").isNull, oldDF.col("userid"))
          .as("userid"),
        //调用自定义函数,将新数据的tag和老数据的tag传入,进行计算.得到最终的标签
        mergeTag(resultDF.col("tagIds"), oldDF.col("tagIds"))
          .as("tagIds")
      )

      //保存数据.
      //保存合并结果.
      saveDF.write
        .format(hbaseSource)
        .option(HBaseMeta.ZKHOSTS, hbase_zkhosts)
        .option(HBaseMeta.ZKPORT, hbase_zkport)
        .option(HBaseMeta.HBASETABLE, hbase_hbasetable)
        .option(HBaseMeta.FAMILY, hbase_family)
        .option(HBaseMeta.SELECTFIELDS, hbase_selectfields)
        .save()
    }
  }


}
