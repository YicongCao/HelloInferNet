using System;
using System.Linq;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Math;

namespace myApp
{
    class TwoCoins
    {
        public static void Run()
        {
            Variable<bool> firstCoin = Variable.Bernoulli(0.5);
            Variable<bool> secondCoin = Variable.Bernoulli(0.5);
            Variable<bool> bothHeads = firstCoin & secondCoin;
            InferenceEngine engine = new InferenceEngine();
            engine.ShowFactorGraph = true;
            Console.WriteLine("Probability both coins are heads: " + engine.Infer(bothHeads));
        }
    }
}