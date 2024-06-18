using System.Diagnostics;

namespace Perceptron
{
    public class Classification
    {
        private List<List<List<double>>> Weights { get; set; } = new List<List<List<double>>>();

        public int InputCount { get; private set; }
        public int OutputCount { get; private set; }
        public List<Func<double, double>> ActivateFunctions { get; private set; }

        public Classification(int inputCount, int[] layer, List<Func<double, double>> activateFunctions)
        {
            InputCount = inputCount;
            OutputCount = layer.Last();
            ActivateFunctions = activateFunctions;

            if (activateFunctions.Count != layer.Length)
            {
                throw new ArgumentException("Activation function's count and layer's count are not samed");
            }

            for (int l = 0; l < layer.Length; l++)
            {
                Weights.Add(new List<List<double>>());

                if (l == 0)
                {
                    for (int p = 0; p < layer[l]; p++)
                    {
                        Weights[l].Add(new List<double>(new double[inputCount]));
                    }
                }
                else
                {
                    for (int p = 0; p < layer[l]; p++)
                    {
                        Weights[l].Add(new List<double>(new double[layer[l - 1]]));
                    }
                }
            }

            for (int l = 0; l < Weights.Count; l++)
            {
                Debug.WriteLine($"L: {l}");

                for (int p = 0; p < Weights[l].Count; p++)
                {
                    Debug.WriteLine($"  P: {p}");

                    for (int w = 0; w < Weights[l][p].Count; w++)
                    {
                        Weights[l][p][w] = new Random().NextDouble();

                        Debug.WriteLine($"    W: {w}");
                        Debug.WriteLine($"    {Weights[l][p][w]} ");
                    }
                }
            }
        }

        private void Run(List<List<List<double>>> copied, double[] input, out double[] output)
        {
            if (input.Length != InputCount)
            {
                throw new ArgumentException("The input's count is not valid");
            }

            double[] before = input;

            for (int l = 0; l < copied.Count; l++)
            {
                double[] next = new double[copied[l].Count];

                for (int p = 0; p < copied[l].Count; p++)
                {
                    double sum = 0;
                    
                    for (int w = 0; w < copied[l][p].Count; w++)
                    {
                        sum += copied[l][p][w] * before[w];
                    }

                    next[p] = ActivateFunctions[l](sum);
                }

                before = next;
            }

            output = before;
        }

        private double GradientDescent(double[] input, double[] output, double omicron, double alpha, int l, int p, int w)
        {
            List<List<List<double>>> copied = new List<List<List<double>>>();

            for (int ll = 0; ll < Weights.Count; ll++)
            {
                copied.Add(new List<List<double>>());

                for (int pp = 0; pp < Weights[ll].Count; pp++)
                {
                    copied[ll].Add(new List<double>());

                    for (int ww = 0; ww < Weights[ll][pp].Count; ww++)
                    {
                        copied[ll][pp].Add(Weights[ll][pp][ww]);
                    }
                }
            }

            double weight = copied[l][p][w]; 

            copied[l][p][w] += omicron;
            Run(copied, input, out double[] oPlus); 
            
            copied[l][p][w] -= 2 * omicron;
            Run(copied, input, out double[] oMinus);

            double errorPlus = 0;
            double errorMinus = 0;

            for (int i = 0; i < output.Length; i++)
            {
                errorPlus += (output[i] - oPlus[i]) * (output[i] - oPlus[i]);
                errorMinus += (output[i] - oMinus[i]) * (output[i] - oMinus[i]);
            }

            double gradient = (errorPlus - errorMinus) / (2 * omicron);
            
            weight -= alpha * gradient;

            return weight;
        }

        private void Epoch(int epoch, double[] input, double[] output, double omicron = 0.01, double alpha = 0.01)
        {
            List<List<List<double>>> newWeights = new List<List<List<double>>>();

            for (int ll = 0; ll < Weights.Count; ll++)
            {
                newWeights.Add(new List<List<double>>());

                for (int pp = 0; pp < Weights[ll].Count; pp++)
                {
                    newWeights[ll].Add(new List<double>());

                    for (int ww = 0; ww < Weights[ll][pp].Count; ww++)
                    {
                        newWeights[ll][pp].Add(Weights[ll][pp][ww]);
                    }
                }
            }

            //Thread[] ts = new Thread[Weights.Count];
            
            for (int ll = Weights.Count - 1; ll >= 0; ll--)
            {
                int l = ll;

                //ts[ll] = new Thread(() =>
                //{
                    for (int p = 0; p < Weights[l].Count; p++)
                    {
                        for (int w = 0; w < Weights[l][p].Count; w++)
                        {
                            newWeights[l][p][w] = GradientDescent(input, output, omicron, alpha, l, p, w);
                            Debug.WriteLine($"Epoch: {epoch}, Layer: {l}, Perceptron: {p}, Weight: {w}, NewWeight: {newWeights[l][p][w]}");
                        }
                    }
                //});
                //ts[ll].Start();
            }

            //for (int l = 0; l < ts.Length; l++)
            //{
            //    ts[l].Join();
            //}

            Weights = newWeights;
        }

        public void Learn(List<double[]> inputs, List<double[]> outputs, int epoch)
        {
            if (inputs.Count != outputs.Count)
            {
                throw new ArgumentException("inputs count and outputs count are not samed");
            }

            for (int i = 0; i < epoch; i++)
            {
                for (int j = 0; j < inputs.Count; j++)
                {
                    Epoch(i, inputs[j], outputs[j]);
                }
            }
        }

        public void Test(double[] input, out double[] output)
        {
            Run(Weights, input, out output);
        }
    }
}
