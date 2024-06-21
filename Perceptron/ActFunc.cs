using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Perceptron
{
    public static class ActFunc
    {
        private const double lambda = 1.0507;
        private const double alpha = 1.67326;

        public static double[] Relu(double[] x)
        {
            double[] y = new double[x.Length];

            for (int i = 0; i < x.Length; i++)
            {
                if (x[i] < 0) y[i] = 0;
            }

            return y;
        }

        public static double[] Sigmoid(double[] x)
        {
            double[] y = new double[x.Length];

            for (int i = 0; i < x.Length; i++)
            {
                y[i] = 1.0 / (1.0 + Math.Exp(-x[i]));
            }

            return y;
        }

        public static double[] Selu(double[] x)
        {
            double[] y = new double[x.Length];

            for (int i = 0; i < x.Length; i++)
            {
                if (x[i] > 0) y[i] = lambda * x[i];
                else y[i] = lambda * (alpha * Math.Exp(x[i]) - alpha);
            }

            return y;
        }

        public static double[] Softmax(double[] input)
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
