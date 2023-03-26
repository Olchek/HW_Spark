package org.apache.spark.ml.made
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{ParamMap, DoubleParam, IntParam}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import org.apache.spark.sql.functions.lit


trait LinearRegressionParams extends HasFeaturesCol with HasLabelCol with HasOutputCol {
  def setFeatureCol(value: String) : this.type = set(featuresCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  val numIterations = new IntParam(this, "numIterations", "Number of iterations")
  val learningRate = new DoubleParam(this, "learningRate", "Learning rate")

  def setNumIters(value: Int): this.type = set(numIterations, value)
  def getNumIters: Int = $(numIterations)

  def setLearningRate(value: Double): this.type = set(learningRate, value)
  def getLearningRate: Double = $(learningRate)

  setDefault(numIterations -> 1000)
  setDefault(learningRate -> 0.01)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())
    SchemaUtils.checkColumnType(schema, getLabelCol, DoubleType)

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, StructField(getOutputCol, DoubleType))
    }
  }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("LinearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {

    // Used to convert untyped dataframes to datasets with vectors
    implicit val encoder : Encoder[Vector] = ExpressionEncoder()

    val assembler = new VectorAssembler()
      .setInputCols(Array("concatenation", $(featuresCol)))
      .setOutputCol("featuresWithConcatenation")

    val datasetWithIntercept  = assembler.transform(dataset.withColumn("concatenation", lit(1.0)))

    val lrData = datasetWithIntercept.select("featuresWithConcatenation", $(labelCol))
      .rdd.map(row => (row.getAs[Vector](0), row.getDouble(1)))
      .collect()

    val dim = lrData(0)._1.size

    var weights = Vectors.zeros(dim).asBreeze.toDenseVector

    for (i <- 0 until $(numIterations)) {
      lrData.grouped(1).foreach { data =>
        val grad = data.foldLeft(Vectors.zeros(dim).asBreeze.toDenseVector) {
          case (sumGrad, (features, label)) =>
            val x = features.asBreeze
            sumGrad + ((x dot weights) - label) * x
        }
        weights -= getLearningRate * grad
      }
    }

    val weightsNoBias = Vectors.fromBreeze(weights(1 until weights.size).toDenseVector)
    val bias = weights(0)

    copyValues(new LinearRegressionModel(
      weightsNoBias,
      bias
    )
    ).setParent(this)

  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](
                           override val uid: String,
                           val weightsNoBias: DenseVector,
                           val bias: Double) extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {


  private[made] def this(weightsNoBias: Vector, bias: Double) =
    this(Identifiable.randomUID("LinearRegressionModel"), weightsNoBias.toDense, bias)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(weightsNoBias, bias), extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val bWeights = weightsNoBias.asBreeze

    val transformUdf = dataset.sqlContext.udf.register(uid + "_prediction",
      (x : Vector) => { (x.asBreeze dot bWeights) + bias})
    dataset.withColumn($(outputCol), transformUdf(dataset($(featuresCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)
      val vectors = weightsNoBias.toArray :+ bias
      sqlContext.createDataFrame(Seq(Tuple1(Vectors.dense(vectors)))).write.parquet(path + "/vectors")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      // Used to convert untyped dataframes to datasets with vectors
      implicit val encoder : Encoder[Vector] = ExpressionEncoder()
      val params = vectors.select(vectors("_1")
        .as[Vector]).first().asBreeze.toDenseVector

      val weightsNoBias = Vectors.fromBreeze(params(0 until params.size - 1))
      val bias = params(params.size - 1)

      val model = new LinearRegressionModel(weightsNoBias, bias)
      metadata.getAndSetParams(model)
      model
    }
  }
}