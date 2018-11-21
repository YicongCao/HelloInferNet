using System;
using System.Linq;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Math;

namespace myApp
{
    class SolveEquation
    {
        public static void Run()
        {
            // Infer.net 能将变量约束限制到一定范围，这个范围可以是一个质点、也可以是一个概率分布
            //Variable<double> r = Variable.GaussianFromMeanAndPrecision(1, 1).Named("radius");
            Variable<double> r = Variable.Random(new Gaussian(1, 1)).Named("radius");
            Variable<double> len = ((Variable<double>)4 * Math.PI).Named("length");
            Variable<double> currlen = ((Variable<double>)2 * r * Math.PI).Named("curr length");
            Variable.ConstrainEqual(currlen, len);
            //Variable.ConstrainBetween(2 * Math.PI * r, len - 0.1, len + 0.1);
            InferenceEngine engine = new InferenceEngine();
            engine.ShowFactorGraph = true;
            var result = engine.Infer(r); // 可以检视一下 result
            Console.WriteLine($"r = {result}");
        }
    }
}