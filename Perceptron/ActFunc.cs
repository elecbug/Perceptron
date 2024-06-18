using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Perceptron
{
    public static class ActFunc
    {
        public static double Relu(double x)
        {
            if (x < 0) return 0;
            return x;
        }

        public static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public static double Selu(double x) 
        {
            if (x < 0) return x / 2;
            return x;
        }
    }
}
