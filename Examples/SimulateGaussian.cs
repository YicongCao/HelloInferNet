using System;
using System.Linq;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Math;

namespace myApp
{
    class SimulateGaussian
    {
        public static void Run()
        {
            InferenceEngine engine = new InferenceEngine();
            double[] data = new double[100];
            while (true)
            {
                for (int i = 0; i < data.Length; i++) data[i] = Rand.Normal(0, 1);
                Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100);
                Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1);
                for (int i = 0; i < data.Length; i++)
                {
                    Variable<double> x = Variable.GaussianFromMeanAndPrecision(mean, precision);
                    x.ObservedValue = data[i];
                }
                Console.WriteLine("mean=" + engine.Infer(mean) + " prec=" + engine.Infer(precision));
            }
        }
    }
}