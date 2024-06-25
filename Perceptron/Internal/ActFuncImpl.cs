using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Perceptron.Enumerable;

namespace Perceptron.Internal
{
    /// <summary>
    /// 활성화 함수 구현
    /// </summary>
    internal static class ActFuncImpl
    {
        /// <summary>
        /// 열거형을 실제 구현으로 변경
        /// </summary>
        /// <param name="act"> 활성화 함수 열거형 </param>
        /// <returns></returns>
        internal static Func<double[], double[]> GetActivation(ActFunc act)
        {
            switch (act)
            {
                case ActFunc.Relu: return Relu;
                case ActFunc.Sigmoid: return Sigmoid;
                case ActFunc.Selu: return Selu;
                case ActFunc.Softmax: return Softmax;
                default: return x => x;
            }
        }

        private const double lambda = 1.0507;
        private const double alpha = 1.67326;

        internal static double[] Relu(double[] x)
        {
            double[] y = new double[x.Length];

            for (int i = 0; i < x.Length; i++)
            {
                if (x[i] < 0) y[i] = 0;
            }

            return y;
        }

        internal static double[] Selu(double[] x)
        {
            double[] y = new double[x.Length];

            for (int i = 0; i < x.Length; i++)
            {
                if (x[i] > 0) y[i] = lambda * x[i];
                else y[i] = lambda * (alpha * Math.Exp(x[i]) - alpha);
            }

            return y;
        }

        internal static double[] Sigmoid(double[] x)
        {
            double[] y = new double[x.Length];

            for (int i = 0; i < x.Length; i++)
            {
                y[i] = 1.0 / (1.0 + Math.Exp(-x[i]));
            }

            return y;
        }

        internal static double[] Softmax(double[] input)
        {
            double max = input[0];
            for (int i = 1; i < input.Length; i++)
            {
                if (input[i] > max)
                {
                    max = input[i];
                }
            }

            double sum = 0.0;
            double[] expValues = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                expValues[i] = Math.Exp(input[i] - max);
                sum += expValues[i];
            }

            double[] softmax = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                softmax[i] = expValues[i] / sum;
            }

            return softmax;
        }
    }
}
