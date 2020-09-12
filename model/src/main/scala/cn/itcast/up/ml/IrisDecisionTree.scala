package cn.itcast.up.ml

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, StringIndexerModel, VectorAssembler}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Author itcast
  * Date 2019/11/5 15:59
  * Desc 
  */
object IrisDecisionTree {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("IrisDecisionTree")
      .master("local[*]")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    import spark.implicits._

    //数据读取
    val source: DataFrame = spark.read
      .csv("file:///D:\\data\\spark\\ml\\iris_tree.csv")
      .toDF("Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width", "Species")
      .select(
        'Sepal_Length cast DoubleType,
        'Sepal_Width cast DoubleType,
        'Petal_Length cast DoubleType,
        'Petal_Width cast DoubleType,
        'Species)
    source.show(false)
    /*
 +------------+-----------+------------+-----------+-----------+
|Sepal_Length|Sepal_Width|Petal_Length|Petal_Width|Species    |
+------------+-----------+------------+-----------+-----------+
|5.1         |3.5        |1.4         |0.2        |Iris-setosa|
|4.9         |3.0        |1.4         |0.2        |Iris-setosa|
|4.7         |3.2        |1.3         |0.2        |Iris-setosa|
|4.6         |3.1        |1.5         |0.2        |Iris-setosa|
|5.0         |3.6        |1.4         |0.2        |Iris-setosa|
|5.4         |3.9        |1.7         |0.4        |Iris-setosa|
|4.6         |3.4        |1.4         |0.3        |Iris-setosa|
|5.0         |3.4        |1.5         |0.2        |Iris-setosa|
|4.4         |2.9        |1.4         |0.2        |Iris-setosa|
|4.9         |3.1        |1.5         |0.1        |Iris-setosa|
|5.4         |3.7        |1.5         |0.2        |Iris-setosa|
|4.8         |3.4        |1.6         |0.2        |Iris-setosa|
|4.8         |3.0        |1.4         |0.1        |Iris-setosa|
|4.3         |3.0        |1.1         |0.1        |Iris-setosa|
|5.8         |4.0        |1.2         |0.2        |Iris-setosa|
|5.7         |4.4        |1.5         |0.4        |Iris-setosa|
|5.4         |3.9        |1.3         |0.4        |Iris-setosa|
|5.1         |3.5        |1.4         |0.3        |Iris-setosa|
|5.7         |3.8        |1.7         |0.3        |Iris-setosa|
|5.1         |3.8        |1.5         |0.3        |Iris-setosa|
+------------+-----------+------------+-----------+-----------+
only showing top 20 rows
     */

    //1.类别处理
    val stringIndexerModel: StringIndexerModel = new StringIndexer()
      .setInputCol("Species")
      .setOutputCol("label").fit(source)

    //2.特征向量化
    val vectorAssembler: VectorAssembler = new VectorAssembler()
      .setInputCols(Array("Sepal_Width", "Petal_Length", "Sepal_Length", "Petal_Width"))
      .setOutputCol("features")

    //3.创建决策树分类器
    val decisionTreeClassification: DecisionTreeClassifier = new DecisionTreeClassifier()
      .setFeaturesCol("features")
      .setPredictionCol("predict")
      .setMaxDepth(5)

    //4.类别还原
    val indexToString: IndexToString = new IndexToString()
      .setInputCol("label")
      .setOutputCol("species_converted")

    //5.划分数据集
    val Array(trainingData, testData) = source.randomSplit(Array(0.8, 0.2),10)

    //6.构建Pipeline
    val pipelineModel: PipelineModel = new Pipeline()
      .setStages(Array(stringIndexerModel, vectorAssembler, decisionTreeClassification, indexToString))
      .fit(trainingData)//训练使用训练集

    //7.预测使用测试集
    val result: DataFrame = pipelineModel.transform(testData)
    result.show(10,false)
/*
+------------+-----------+------------+-----------+---------------+-----+-----------------+--------------+-------------+-------+-----------------+
|Sepal_Length|Sepal_Width|Petal_Length|Petal_Width|Species        |label|features         |rawPrediction |probability  |predict|species_converted|
+------------+-----------+------------+-----------+---------------+-----+-----------------+--------------+-------------+-------+-----------------+
|4.4         |3.0        |1.3         |0.2        |Iris-setosa    |0.0  |[3.0,1.3,4.4,0.2]|[37.0,0.0,0.0]|[1.0,0.0,0.0]|0.0    |Iris-setosa      |
|4.6         |3.6        |1.0         |0.2        |Iris-setosa    |0.0  |[3.6,1.0,4.6,0.2]|[37.0,0.0,0.0]|[1.0,0.0,0.0]|0.0    |Iris-setosa      |
|5.0         |3.3        |1.4         |0.2        |Iris-setosa    |0.0  |[3.3,1.4,5.0,0.2]|[37.0,0.0,0.0]|[1.0,0.0,0.0]|0.0    |Iris-setosa      |
|5.0         |3.4        |1.5         |0.2        |Iris-setosa    |0.0  |[3.4,1.5,5.0,0.2]|[37.0,0.0,0.0]|[1.0,0.0,0.0]|0.0    |Iris-setosa      |
|5.0         |3.5        |1.6         |0.6        |Iris-setosa    |0.0  |[3.5,1.6,5.0,0.6]|[37.0,0.0,0.0]|[1.0,0.0,0.0]|0.0    |Iris-setosa      |
|5.0         |3.6        |1.4         |0.2        |Iris-setosa    |0.0  |[3.6,1.4,5.0,0.2]|[37.0,0.0,0.0]|[1.0,0.0,0.0]|0.0    |Iris-setosa      |
|5.1         |3.8        |1.5         |0.3        |Iris-setosa    |0.0  |[3.8,1.5,5.1,0.3]|[37.0,0.0,0.0]|[1.0,0.0,0.0]|0.0    |Iris-setosa      |
|5.2         |3.4        |1.4         |0.2        |Iris-setosa    |0.0  |[3.4,1.4,5.2,0.2]|[37.0,0.0,0.0]|[1.0,0.0,0.0]|0.0    |Iris-setosa      |
|5.2         |3.5        |1.5         |0.2        |Iris-setosa    |0.0  |[3.5,1.5,5.2,0.2]|[37.0,0.0,0.0]|[1.0,0.0,0.0]|0.0    |Iris-setosa      |
|5.4         |3.0        |4.5         |1.5        |Iris-versicolor|1.0  |[3.0,4.5,5.4,1.5]|[0.0,40.0,0.0]|[0.0,1.0,0.0]|1.0    |Iris-versicolor  |
+------------+-----------+------------+-----------+---------------+-----+-----------------+--------------+-------------+-------+-----------------+
only showing top 10 rows
 */
    //8.模型评价
    val evaluator: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("predict")
      .setMetricName("accuracy")//使用准确(性)评价
    //评估准确(性)
    val accuracy: Double = evaluator.evaluate(result)
    println("模型的准确率为:"+accuracy)
    //模型的准确率为:0.9736842105263158

    //9.查看决策树的决策过程
    //通过DecisionTreeClassifier的源码发现DecisionTreeClassifier和DecisionTreeClassificationModel存在"间接继承"关系
    val model: DecisionTreeClassificationModel = pipelineModel.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    println("决策树的决策过程为:\n"+model.toDebugString)//获取决策过程
/*
DecisionTreeClassificationModel (uid=dtc_9a9fcdcee3f9) of depth 5 with 15 nodes
  If (feature 1 <= 1.9)
   Predict: 0.0
  Else (feature 1 > 1.9)
   If (feature 1 <= 4.9)
    If (feature 3 <= 1.6)
     Predict: 1.0
    Else (feature 3 > 1.6)
     If (feature 0 <= 3.0)
      Predict: 2.0
     Else (feature 0 > 3.0)
      Predict: 1.0
   Else (feature 1 > 4.9)
    If (feature 2 <= 6.0)
     If (feature 2 <= 5.9)
      Predict: 2.0
     Else (feature 2 > 5.9)
      If (feature 0 <= 2.2)
       Predict: 2.0
      Else (feature 0 > 2.2)
       Predict: 1.0
    Else (feature 2 > 6.0)
     Predict: 2.0
 */
  }
}
