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

        public static double Relu(double x)
        {
            if (x < 0) return 0;
            else return x;
        }

        public static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public static double Selu(double x)
        {
            if (x > 0) return lambda * x;
            else return lambda * (alpha * Math.Exp(x) - alpha);
        }
    }
}
