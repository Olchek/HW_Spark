package org.apache.spark.ml.made

import scala.collection.JavaConverters._
import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{VectorUDT, Vectors}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row}
import breeze.linalg._

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 0.01
  lazy val data: DataFrame = LinearRegressionTest._data

  "Estimator" should "calculate weights and bias" in {
    val estimator = new LinearRegression()
      .setFeatureCol("features")
      .setLabelCol("label")
      .setOutputCol("predictions")

    val model = estimator.fit(data)

    validateModel(model, model.transform(data))
  }

  "Estimator" should "not learn with zero iterations" in {
    val estimator = new LinearRegression()
      .setFeatureCol("features")
      .setLabelCol("label")
      .setOutputCol("predictions")

    estimator.setNumIters(0)

    val model = estimator.fit(data)

    model.weightsNoBias(0) should be(0.0 +- delta)
    model.weightsNoBias(1) should be(0.0 +- delta)
    model.weightsNoBias(2) should be(0.0 +- delta)
    model.bias should be(0.0 +- delta)
  }

  "Estimator" should "not learn with negative learning rate" in {
    val estimator = new LinearRegression()
      .setFeatureCol("features")
      .setLabelCol("label")
      .setOutputCol("predictions")

    estimator.setLearningRate(-0.01)

    val model = estimator.fit(data)

    model.weightsNoBias(0) should not be(1.5 +- delta)
    model.weightsNoBias(1) should not be(0.3 +- delta)
    model.weightsNoBias(2) should not be(-0.7 +- delta)
    model.bias should not be(-2.0 +- delta)
  }
  "Estimator" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeatureCol("features")
        .setLabelCol("label")
        .setOutputCol("predictions")
    ))

    val tmpFolder = Files.createTempDir()
    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)
    val model = reRead.fit(data).stages(0).asInstanceOf[LinearRegressionModel]

    validateModel(model, model.transform(data))
  }

  "Model" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeatureCol("features")
        .setLabelCol("label")
        .setOutputCol("predictions")
    ))

    val model = pipeline.fit(data)
    val tmpFolder = Files.createTempDir()
    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)
    validateModel(model.stages(0).asInstanceOf[LinearRegressionModel], reRead.transform(data))
  }

  private def validateModel(model: LinearRegressionModel, data: DataFrame) = {
    val predictions: Array[Double] = data.collect().map(_.getAs[Double](2))

    model.weightsNoBias(0) should be(1.5 +- delta)
    model.weightsNoBias(1) should be(0.3 +- delta)
    model.weightsNoBias(2) should be(-0.7 +- delta)
    model.bias should be(-2.0 +- delta)

    predictions.length should be(data.count())
  }
}

object LinearRegressionTest extends WithSpark {

  lazy val schema: StructType = StructType(
    Array(
      StructField("label", DoubleType),
      StructField("features", new VectorUDT())
    ))

  lazy val matrix = DenseMatrix.rand(1000, 3)
  lazy val trueWeigthsNoBias = DenseVector(1.5, 0.3, -0.7)
  lazy val trueBias = -2.0
  lazy val labels = matrix * trueWeigthsNoBias + trueBias

  lazy val rowData = (0 until labels.length).map(i => Row(labels(i), Vectors.dense(matrix(i, ::).t.toArray)))

  lazy val _data: DataFrame = sqlc.createDataFrame(rowData.asJava, schema)
}