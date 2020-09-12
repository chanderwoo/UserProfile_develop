package cn.itcast.up.ml

import java.util.Date

import cn.itcast.up.base.BaseModel
import cn.itcast.up.bean.HBaseMeta
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.types.DoubleType

/**
  * Author itcast
  * Date 2019/11/6 11:36
  * Desc 使用决策树完成用户购物性别模型
  */
object USGModel extends BaseModel{
  def main(args: Array[String]): Unit = {
    execute()
  }

  /**
    * 获取标签id(即模型id,该方法应该在编写不同模型时进行实现)
    * @return
    */
  override def getTagID(): Int = 56

  /**
    * 开始计算
    * @param fiveDF  MySQL中的5级规则 id,rule
    * @param hbaseDF 根据selectFields查询出来的HBase中的数据
    * @return userid,tagIds
    */
  override def compute(fiveDF: DataFrame, hbaseDF: DataFrame): DataFrame = {
    import org.apache.spark.sql.functions._
    import spark.implicits._
    //1.加载数据
    //fiveDF.show(10,false)
    //fiveDF.printSchema()
    /*
 +---+----+
|id |rule|
+---+----+
|57 |0   |
|58 |1   |
|59 |-1  |
+---+----+
     */

    val goodsDF: DataFrame = hbaseDF
    //goodsDF.show(10,false)
    //goodsDF.printSchema()
    /*
 +----------------------+---------+-----------+
|cOrderSn              |ogColor  |productType|
+----------------------+---------+-----------+
|jd_14091818005983607  |白色       |烤箱         |
|jd_14091317283357943  |香槟金      |冰吧         |
|jd_14092012560709235  |香槟金色     |净水机        |
|rrs_15234137          |梦境极光【布朗灰】|烤箱         |
|suning_790750687478116|梦境极光【卡其金】|4K电视       |
|rsq_805093707860210   |黑色       |烟灶套系       |
|jd_14090910361908941  |黑色       |智能电视       |
|jd_14091823464864679  |香槟金色     |燃气灶        |
|jd_14091817311906413  |银色       |滤芯         |
|suning_804226647488814|玫瑰金      |电饭煲        |
+----------------------+---------+-----------+
only showing top 10 rows
     */

    val ordersDF: DataFrame = spark.read
      .format("cn.itcast.up.tools.HBaseSource")
      .option(HBaseMeta.ZKHOSTS, "bd001")
      .option(HBaseMeta.ZKPORT, "2181")
      .option(HBaseMeta.HBASETABLE, "tbl_orders")
      .option(HBaseMeta.FAMILY, "detail")
      .option(HBaseMeta.SELECTFIELDS, "memberId,orderSn")
      .load()
    //ordersDF.show(10,false)
    //ordersDF.printSchema()
    /*
 +---------+-------------------+
|memberId |orderSn            |
+---------+-------------------+
|13823431 |ts_792756751164275 |
|4035167  |D14090106121770839 |
|4035291  |D14090112394810659 |
|4035041  |fx_787749561729045 |
|13823285 |D14092120154435903 |
|4034219  |D14092120155620305 |
|138230939|top_810791455519102|
|4035083  |D14092120161884409 |
|138230935|D14092120162313538 |
|13823231 |D14092120162378713 |
+---------+-------------------+
     */

    //2.特征选取(实际中可以选取商品的众多属性再进行主成份分析PCA降维,我们这里考虑到时间原因,只选取2个)
    //颜色ID应该来源于字典表,这里简化处理
    val color: Column = functions
      .when('ogColor.equalTo("银色"), 1)
      .when('ogColor.equalTo("香槟金色"), 2)
      .when('ogColor.equalTo("黑色"), 3)
      .when('ogColor.equalTo("白色"), 4)
      .when('ogColor.equalTo("梦境极光【卡其金】"), 5)
      .when('ogColor.equalTo("梦境极光【布朗灰】"), 6)
      .when('ogColor.equalTo("粉色"), 7)
      .when('ogColor.equalTo("金属灰"), 8)
      .when('ogColor.equalTo("金色"), 9)
      .when('ogColor.equalTo("乐享金"), 10)
      .when('ogColor.equalTo("布鲁钢"), 11)
      .when('ogColor.equalTo("月光银"), 12)
      .when('ogColor.equalTo("时尚光谱【浅金棕】"), 13)
      .when('ogColor.equalTo("香槟色"), 14)
      .when('ogColor.equalTo("香槟金"), 15)
      .when('ogColor.equalTo("灰色"), 16)
      .when('ogColor.equalTo("樱花粉"), 17)
      .when('ogColor.equalTo("蓝色"), 18)
      .when('ogColor.equalTo("金属银"), 19)
      .when('ogColor.equalTo("玫瑰金"), 20)
      .otherwise(0)
      .alias("color")
    //类型ID应该来源于字典表,这里简化处理
    val productType: Column = functions
      .when('productType.equalTo("4K电视"), 9)
      .when('productType.equalTo("Haier/海尔冰箱"), 10)
      .when('productType.equalTo("Haier/海尔冰箱"), 11)
      .when('productType.equalTo("LED电视"), 12)
      .when('productType.equalTo("Leader/统帅冰箱"), 13)
      .when('productType.equalTo("冰吧"), 14)
      .when('productType.equalTo("冷柜"), 15)
      .when('productType.equalTo("净水机"), 16)
      .when('productType.equalTo("前置过滤器"), 17)
      .when('productType.equalTo("取暖电器"), 18)
      .when('productType.equalTo("吸尘器/除螨仪"), 19)
      .when('productType.equalTo("嵌入式厨电"), 20)
      .when('productType.equalTo("微波炉"), 21)
      .when('productType.equalTo("挂烫机"), 22)
      .when('productType.equalTo("料理机"), 23)
      .when('productType.equalTo("智能电视"), 24)
      .when('productType.equalTo("波轮洗衣机"), 25)
      .when('productType.equalTo("滤芯"), 26)
      .when('productType.equalTo("烟灶套系"), 27)
      .when('productType.equalTo("烤箱"), 28)
      .when('productType.equalTo("燃气灶"), 29)
      .when('productType.equalTo("燃气热水器"), 30)
      .when('productType.equalTo("电水壶/热水瓶"), 31)
      .when('productType.equalTo("电热水器"), 32)
      .when('productType.equalTo("电磁炉"), 33)
      .when('productType.equalTo("电风扇"), 34)
      .when('productType.equalTo("电饭煲"), 35)
      .when('productType.equalTo("破壁机"), 36)
      .when('productType.equalTo("空气净化器"), 37)
      .otherwise(0)
      .alias("productType")

    //3.数据标注(根据运营的前期的统计分析对数据进行性别类别的标注)
    //目标是根据运营规则标注的部分数据,进行决策树模型的训练,以后来了新的数据,模型就可以判断出购物性别是男还是女
    val label: Column = functions
      .when('ogColor.equalTo("樱花粉")
        .or('ogColor.equalTo("白色"))
        .or('ogColor.equalTo("香槟色"))
        .or('ogColor.equalTo("香槟金"))
        .or('productType.equalTo("料理机"))
        .or('productType.equalTo("挂烫机"))
        .or('productType.equalTo("吸尘器/除螨仪")), 1) //女
      .otherwise(0)//男
      .alias("gender")//决策树预测label

    //4.将数据表进行合并
    //最终需要找到用户和用户所购买的所有商品,进行训练,找到商品和性别之间的关系
    val source = goodsDF.select('cOrderSn as "orderSn", color, productType, label)
      .join(ordersDF, "orderSn")
      .select('memberId as "userId", 'color, 'productType, 'gender)
    //source.show(10,false)
/*
+---------+-----+-----------+------+
|userId   |color|productType|gender|
+---------+-----+-----------+------+
|13823535 |16   |0          |0     |
|13823535 |1    |24         |0     |
|13823535 |7    |30         |0     |
|13823391 |10   |14         |0     |
|4034493  |9    |12         |0     |
|13823683 |8    |17         |0     |
|62       |9    |15         |0     |
|4035201  |8    |12         |0     |
|13823449 |10   |0          |0     |
|138230919|12   |15         |0     |
+---------+-----+-----------+------+
 */
    //机器学习部分
    //1.类别处理
    val stringIndexerModel: StringIndexerModel = new StringIndexer()
    .setInputCol("gender")
    .setOutputCol("label").fit(source)

    //2.特征向量化
    val vectorAssembler: VectorAssembler = new VectorAssembler()
      .setInputCols(Array("color", "productType"))
      .setOutputCol("features")
    val featureDF: DataFrame = vectorAssembler.transform(source)

    //3.对特征进行索引,大于3个不同的值的特征被视为连续特征
    //VectorIndexer是对数据集特征向量中的类别(离散值)特征(index categorical features categorical features)进行编号。
    //它能够自动判断那些特征是离散值型的特征，并对他们进行编号，具体做法是通过设置一个maxCategories，
    //特征向量中某一个特征不重复取值个数小于maxCategories，则被重新编号为0～K（K<=maxCategories-1）。
    //某一个特征不重复取值个数大于maxCategories，则该特征视为连续值，不会重新编号（不会发生任何改变）
    //主要作用：提高决策树或随机森林等ML方法的分类效果
    val featureVectorIndexer: VectorIndexerModel = new VectorIndexer()
     .setInputCol("features")
     .setOutputCol("featureIndexed")
     .setMaxCategories(3)
     .fit(featureDF)

    //4.构建分类器
    val decisionTreeClassification: DecisionTreeClassifier = new DecisionTreeClassifier()
      .setFeaturesCol("featureIndexed")
      .setPredictionCol("predict")
      .setMaxDepth(5)
      //.setImpurity("gini")

    //5.类别还原
    val indexToString: IndexToString = new IndexToString()
      .setInputCol("label")
      .setOutputCol("gender_converted")

    //6.数据集划分
    val Array(trainingData, testData) = source.randomSplit(Array(0.8,0.2))

    //7.构建Pipeline并训练模型
    val pipeline: Pipeline = new Pipeline().setStages(Array(stringIndexerModel,vectorAssembler,featureVectorIndexer,decisionTreeClassification,indexToString))
    val pipelineModel: PipelineModel = pipeline.fit(trainingData)

    //8.预测
    val testReusltDF: DataFrame = pipelineModel.transform(testData)
    val trainingReusltDF: DataFrame = pipelineModel.transform(trainingData)

    //9.决策过程输出
    val classificationModel: DecisionTreeClassificationModel = pipelineModel.stages(3).asInstanceOf[DecisionTreeClassificationModel]
    println("决策过程如下:\n"+classificationModel.toDebugString)

    //10.评价
    this.evaluateACCAndAUC(trainingReusltDF,testReusltDF)


    //11.返回结果并保存
    val allResult: Dataset[Row] = testReusltDF.union(trainingReusltDF)
    val tempDF: DataFrame = allResult.select('userId,
      when('predict === 0, 1).otherwise(0).as("male"), //计算每个用户所有订单中的男性商品的订单数
      when('predict === 1, 1).otherwise(0).as("female")) //计算每个用户所有订单中的女性商品的订单数
      .groupBy('userId)
      .agg(
        count('userId) cast DoubleType as "total", //总共预测多少次
        sum('male) cast DoubleType as "maleCount", //预测为男的次数
        sum('female) cast DoubleType as "femaleCount") //预测为女的次数
      .select('userId, 'total, 'maleCount, 'femaleCount)
    tempDF.show(10,false)
/*
+---------+-----+---------+-----------+
|userId   |total|maleCount|femaleCount|
+---------+-----+---------+-----------+
|4033473  |13.0 |12.0     |1.0        |
|13823083 |17.0 |13.0     |4.0        |
|13823681 |3.0  |3.0      |0.0        |
|138230919|5.0  |3.0      |2.0        |
|13822725 |7.0  |6.0      |1.0        |
|4033575  |9.0  |8.0      |1.0        |
|4034191  |6.0  |5.0      |1.0        |
|13823153 |9.0  |6.0      |3.0        |
|13822841 |3.0  |3.0      |0.0        |
|13823431 |8.0  |4.0      |4.0        |
+---------+-----+---------+-----------+
 */

    //12.上面我们就得到了用户在该网站被预测购物性别的总次数,以及被预测为男和女各多少次
    //接下来我们要结合该用户所有的预测结果真正的给该用户打上购物性别标签
    //预测规则A:每个订单的男性商品>=80%则认定为该订单的用户为男，或女商品比例达到80%则认定为该订单的用户为女；
    //由于是家电产品，一个订单中通常只有一个商品。调整规则A为规则B：
    //预测规则B:计算每个用户近半年内所有订单中的男性商品超过60%则认定该用户为男，或近半年内所有订单中的女性品超过60%则认定该用户为女
    //那么现在我们要做的事情可以简化为:根据该用户的total/maleCount/femaleCount判断用户应该被打上哪个标签
    //12.1 将5级标签转为map方便后面使用
    /*
+---+----+
|id |rule|
+---+----+
|57 |0   |
|58 |1   |
|59 |-1  |
+---+----+
     */
    //Map[rule, tagId]
    val ruleMap: Map[String, Long] = fiveDF.collect().map(row=>(row.getString(1),row.getLong(0))).toMap


    val getGenderTag = udf((total:Double,maleCount:Double,femaleCount:Double)=>{
      //预测规则B:计算每个用户近半年内所有订单中的男性商品超过60%则认定该用户为男，或近半年内所有订单中的女性品超过60%则认定该用户为女
      val maleRate: Double = maleCount / total
      val femaleRate: Double = femaleCount/ total
      if (maleRate>= 0.6){
        ruleMap("0")
      }else if(femaleRate>=0.6){
        ruleMap("1")
      }else{
        ruleMap("-1")
      }
    })

    //12.2使用自定义UDF完成tag转换
    val newDF: DataFrame = tempDF.select('userId,getGenderTag('total,'maleCount,'femaleCount).as("tagIds"))
    newDF.show(10,false)

    println(new Date().toLocaleString)//需要20多分钟
    newDF
  }
  /**
    * @param predictTestDF
    * @param predictTrainDF
    */
  def evaluateACCAndAUC(predictTrainDF: DataFrame,predictTestDF: DataFrame): Unit = {
    // 1. ACC
    val accEvaluator = new MulticlassClassificationEvaluator()
      .setPredictionCol("predict")
      .setLabelCol("label")
      .setMetricName("accuracy")//精准度

    val trainAcc: Double = accEvaluator.evaluate(predictTrainDF)
    val testAcc: Double = accEvaluator.evaluate(predictTestDF)
    println(s"训练集上的 ACC 是 : $trainAcc")
    println(s"测试集上的 ACC 是 : $testAcc")
    //训练集上的 ACC 是 : 0.9659155462980497
    //测试集上的 ACC 是 : 0.9666924864446166

    // 2. AUC
    val trainRdd: RDD[(Double, Double)] = predictTrainDF.select("label", "predict").rdd
      .map(row => (row.getAs[Double](0), row.getAs[Double](1)))
    val testRdd: RDD[(Double, Double)] = predictTestDF.select("label", "predict").rdd
      .map(row => (row.getAs[Double](0), row.getAs[Double](1)))

    val trainAUC: Double = new BinaryClassificationMetrics(trainRdd).areaUnderROC()
    val testAUC: Double = new BinaryClassificationMetrics(testRdd).areaUnderROC()
    println(s"训练集上的 AUC 是 : $trainAUC")
    println(s"测试集上的 AUC 是 : $testAUC")
    //训练集上的 AUC 是 : 0.955163396745856
    //测试集上的 AUC 是 : 0.955110198789974

  }
}
