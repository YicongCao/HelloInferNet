using System;
using System.Linq;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;

namespace myApp
{
    class OnlineLearning
    {
        public static void Run()
        {
            Variable<int> nItems = Variable.New<int>().Named("nItems");
            Range item = new Range(nItems).Named("item");
            Variable<double> variance = Variable.GammaFromShapeAndRate(1.0, 1.0).Named("variance");
            Variable<double> mean = Variable.GaussianFromMeanAndVariance(0.0, variance).Named("mean");
            VariableArray<double> x = Variable.Array<double>(item).Named("x");
            x[item] = Variable.GaussianFromMeanAndPrecision(mean, 1.0).ForEach(item);

            mean.AddAttribute(QueryTypes.Marginal);
            mean.AddAttribute(QueryTypes.MarginalDividedByPrior);
            InferenceEngine engine = new InferenceEngine();
            engine.ShowFactorGraph = true;

            // inference on a single batch  
            double[] data = new double[100];
            for (int i = 0; i < data.Length; i++) data[i] = Rand.Normal(1, 1);
            x.ObservedValue = data;
            nItems.ObservedValue = data.Length;
            Gaussian meanExpected = engine.Infer<Gaussian>(mean);

            // online learning in mini-batches  
            Variable<Gaussian> meanMessage = Variable.Observed<Gaussian>(Gaussian.Uniform()).Named("meanMessage");
            Variable.ConstrainEqualRandom(mean, meanMessage);

            int batchSize = 1;
            double[][] dataBatches = new double[data.Length / batchSize][];
            for (int batch = 0; batch < dataBatches.Length; batch++)
            {
                dataBatches[batch] = data.Skip(batch * batchSize).Take(batchSize).ToArray();
            }
            Gaussian meanMarginal = Gaussian.Uniform();
            for (int batch = 0; batch < dataBatches.Length; batch++)
            {
                nItems.ObservedValue = dataBatches[batch].Length;
                x.ObservedValue = dataBatches[batch];
                meanMarginal = engine.Infer<Gaussian>(mean);
                Console.WriteLine("mean after batch {0} = {1}, meanMsg = {2}", batch, meanMarginal, meanMessage.ObservedValue);
                meanMessage.ObservedValue = engine.Infer<Gaussian>(mean, QueryTypes.MarginalDividedByPrior);
                engine.ShowFactorGraph = true;
            }
            // the answers should be identical for this simple model  
            Console.WriteLine("mean = {0} should be {1}", meanMarginal, meanExpected);
            Console.WriteLine("variance = {0}", engine.Infer(variance));
        }
    }
}