package cn.itcast.up.ml

import cn.itcast.up.base.BaseModel
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.{DataFrame, Dataset, Row}

/**
  * Author itcast
  * Date 2019/11/8 9:26
  * Desc 使用基于隐语义模型的ALS推荐算法,构建用户购物偏好模型(商品/品牌偏好模型)
  */
object BPModel extends BaseModel{
  def main(args: Array[String]): Unit = {
    execute()
  }
  /**
    * 获取标签id(即模型id,该方法应该在编写不同模型时进行实现)
    * @return
    */
  override def getTagID(): Int = 60

  /**
    * 开始计算
    * @param fiveDF  MySQL中的5级规则 id,rule
    * @param hbaseDF 根据selectFields查询出来的HBase中的数据
    * @return userid,tagIds
    */
  override def compute(fiveDF: DataFrame, hbaseDF: DataFrame): DataFrame = {
    //hbaseDF.show(10,false)
    //hbaseDF.printSchema()
    /*
+--------------+-------------------------------------------------------------------+-------------------+
|global_user_id|loc_url                                                            |log_time           |
+--------------+-------------------------------------------------------------------+-------------------+
|424           |http://m.eshop.com/mobile/coupon/getCoupons.html?couponsId=3377    |2019-08-13 03:03:55|
|619           |http://m.eshop.com/?source=mobile                                  |2019-07-29 15:07:41|
|898           |http://m.eshop.com/mobile/item/11941.html                          |2019-08-14 09:23:44|
|642           |http://www.eshop.com/l/2729-2931.html                              |2019-08-11 03:20:17|
|130           |http://www.eshop.com/product/5552.html?ebi=ref-lst-1-1             |2019-08-12 11:59:28|
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

    import org.apache.spark.sql.functions._
    import spark.implicits._


    //1.从数据源中查询用户id,浏览的商品id
    val url2productId = udf((url: String) => {
      var productId: String = null
      if (url.contains("/product/") && url.contains(".html")) {
        val start: Int = url.indexOf("/product/")
        val end: Int = url.indexOf(".html")
        if (end > start) {
          productId = url.substring(start + 9, end)
        }
      }
      productId
    })

    val tempDF: Dataset[Row] = hbaseDF.select('global_user_id as "userId",url2productId('loc_url).as("productId"),'log_time).filter('productId.isNotNull)
    //tempDF.show(10,false)
    //tempDF.printSchema()
    /*
+------+---------+-------------------+
|userId|productId|log_time           |
+------+---------+-------------------+
|81    |11013    |2019-08-06 09:10:37|
|767   |11813    |2019-07-28 18:53:10|
|302   |5353     |2019-07-24 09:22:44|
|370   |9221     |2019-08-07 10:18:10|
|405   |4167     |2019-08-04 12:58:18|
|685   |9763     |2019-08-10 21:10:13|
|733   |9501     |2019-08-05 04:17:50|
|659   |11457    |2019-07-29 05:06:50|
|642   |12231    |2019-08-09 07:26:11|
|182   |9763     |2019-07-24 22:08:51|
+------+---------+-------------------+
only showing top 10 rows

root
 |-- userId: string (nullable = true)
 |-- productId: string (nullable = true)
 |-- log_time: string (nullable = true)
     */

    //2.查询用户id(Int),浏览的商品id(Int),浏览次数(评分)(Double)
    val ratingDF: DataFrame = tempDF.groupBy('userId, 'productId)
      .agg(count('productId).as("rating"), first('log_time).as("log_time"))
      .select('userId.cast(DataTypes.IntegerType), 'productId.cast(DataTypes.IntegerType), 'rating.cast(DataTypes.DoubleType))
      //ALS only supports values in Integer range for columns userId and productId. Value null was not numeric.
      .filter('userId.isNotNull && 'productId.isNotNull && 'rating.isNotNull)
    //ratingDF.show(10,false)
    //ratingDF.printSchema()
    /*
 +------+---------+------+
|userId|productId|rating|
+------+---------+------+
|533   |11455    |1.0   |
|322   |11949    |1.0   |
|258   |7467     |1.0   |
|558   |10937    |1.0   |
|555   |10333    |1.0   |
|24    |11111    |1.0   |
|601   |5214     |2.0   |
|756   |10795    |1.0   |
|501   |12233    |3.0   |
|395   |9499     |1.0   |
+------+---------+------+
only showing top 10 rows

root
 |-- userId: integer (nullable = true)
 |-- productId: integer (nullable = true)
 |-- rating: double (nullable = false)
     */

    //使用ALS算法训练模型并预测用户可能感兴趣的商品
    //训练
    val model: ALSModel = new ALS()
      .setUserCol("userId") //用户id列
      .setItemCol("productId") //商品id列
      .setRatingCol("rating") //评分列
      .setPredictionCol("predict")
      .setImplicitPrefs(true) //我们使用的是浏览次数当作评分,所以是隐式评分
      //冷启动策略支持nan和drop，采用drop //https://www.jianshu.com/p/182ae2ceb1d3
      .setColdStartStrategy("drop")
      .setAlpha(1.0) //适用于ALS的隐式反馈变量的参数，其控制偏好观察中的基线置信度（默认为1.0）
      .setMaxIter(2) //要运行的最大迭代次数（默认为10）
      .setRank(2) //模型中潜在因子的数量（默认为10）
      .setRegParam(1.0) //指定ALS中的正则化参数（默认为1.0）
      .fit(ratingDF)


    //预测,给所有用户推荐他可能感兴趣的5件商品
    val result: DataFrame = model.recommendForAllUsers(5)
    result.show(10,false)
    ratingDF.printSchema()
    /*
+------+------------------------------------------------------------------------------------------------------+
|userId|recommendations                                                                                       |
+------+------------------------------------------------------------------------------------------------------+
|471   |[[10935,0.010896015], [6603,0.010697873], [9371,0.010519041], [6395,0.009751318], [7173,0.00967749]]  |
|463   |[[10935,0.010526508], [6603,0.01032451], [9371,0.010150487], [6395,0.009417808], [7173,0.009351814]]  |
|833   |[[10935,0.010327509], [6603,0.010139376], [9371,0.009969836], [6395,0.00924245], [7173,0.009172639]]  |
|496   |[[10935,0.010160314], [6603,0.009975145], [9371,0.009808339], [6395,0.009092798], [7173,0.00902416]]  |
|148   |[[10935,0.009785844], [6603,0.0096077975], [9371,0.009447174], [6395,0.008757753], [7173,0.008691493]]|
|540   |[[10935,0.01053743], [6603,0.010358198], [9371,0.010186721], [6395,0.009433712], [7173,0.009356068]]  |
|392   |[[10935,0.010595391], [6603,0.010400847], [9371,0.010226726], [6395,0.009481777], [7173,0.009410929]] |
|243   |[[10935,0.011122545], [6603,0.010914978], [9371,0.010731797], [6395,0.009952633], [7173,0.009879945]] |
|623   |[[10935,0.010162992], [6603,0.009974668], [9371,0.009807449], [6395,0.009094367], [7173,0.009027276]] |
|737   |[[10935,0.010941938], [6603,0.010745175], [9371,0.010565853], [6395,0.009793008], [7173,0.009717753]] |
+------+------------------------------------------------------------------------------------------------------+
only showing top 10 rows

root
 |-- userId: integer (nullable = true)
 |-- productId: integer (nullable = true)
 |-- rating: double (nullable = false)
     */


    null
  }
}



