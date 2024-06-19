using System.Diagnostics;
using System.Text.Json;

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
                        Weights[l].Add(new List<double>(new double[inputCount + 1]));
                    }
                }
                else
                {
                    for (int p = 0; p < layer[l]; p++)
                    {
                        Weights[l].Add(new List<double>(new double[layer[l - 1] + 1]));
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
                    
                    for (int w = 0; w < copied[l][p].Count - 1; w++)
                    {
                        sum += copied[l][p][w] * before[w];
                    }

                    sum += copied[l][p].Last();

                    next[p] = ActivateFunctions[l](sum);
                }

                before = next;
            }

            output = before;
        }

        private double GradientDescent(double[] input, double[] output, double omicron, double alpha, int jump, int l, int p, int w)
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

            for (int i = 0; i < jump; i++)
            {
                copied[l][p][w] += omicron;
                Run(copied, input, out double[] oPlus);

                copied[l][p][w] -= 2 * omicron;
                Run(copied, input, out double[] oMinus);

                double errorPlus = 0;
                double errorMinus = 0;

                for (int o = 0; o < output.Length; o++)
                {
                    errorPlus += (output[o] - oPlus[o]) * (output[o] - oPlus[o]);
                    errorMinus += (output[o] - oMinus[o]) * (output[o] - oMinus[o]);
                }

                double gradient = (errorPlus - errorMinus) / (2 * omicron);

                weight -= alpha * gradient;

                copied[l][p][w] = weight;
            }

            return weight;
        }

        /* private void ThreadEpoch(int epoch, double[] input, double[] output, double omicron, double alpha, int jump, int maxCount)
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

            Thread[] ts = new Thread[maxCount];
            object[] locker = new object[maxCount];
            (bool, int, int, int)[] parameters = new (bool, int, int, int)[maxCount];

            for (int i = 0; i < ts.Length; i++)
            {
                locker[i] = new object();

                int x = i;
                ts[i] = new Thread(() =>
                {
                    while (true)
                    {
                        while (parameters[x].Item1 == false) ;

                        lock (locker[x])
                        {
                            int l = parameters[x].Item2, p = parameters[x].Item3, w = parameters[x].Item4;

                            newWeights[l][p][w] = GradientDescent(input, output, omicron, alpha, jump, l, p, w);
                            Debug.WriteLine($"Epoch: {epoch}, Layer: {l}, Perceptron: {p}, Weight: {w}, NewWeight: {newWeights[l][p][w]}");

                            parameters[x].Item1 = false;
                        }
                    }
                });

                ts[i].Start();
            }

            for (int l = Weights.Count - 1; l >= 0; l--)
            {
                for (int p = 0; p < Weights[l].Count; p++)
                {
                    for (int w = 0; w < Weights[l][p].Count; w++)
                    {
                        for (int i = 0; i < locker.Length; i++)
                        {
                            if (parameters[i].Item1 == false)
                            {
                                lock (locker[i])
                                {
                                    parameters[i] = (true, l, p, w);
                                }

                                break;
                            }
                            if (i == locker.Length - 1)
                            {
                                i = -1;
                            }
                        }
                    }
                }

                for (int i = 0; i < locker.Length; i++)
                {
                    while (parameters[i].Item1 == true) ;
                }
             
                Weights = newWeights;
            }
        } */

        private void Epoch(int epoch, double[] input, double[] output, double omicron, double alpha, int jump)
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

            for (int l = Weights.Count - 1; l >= 0; l--)
            {
                for (int p = 0; p < Weights[l].Count; p++)
                {

                    for (int w = 0; w < Weights[l][p].Count; w++)
                    {
                        newWeights[l][p][w] = GradientDescent(input, output, omicron, alpha, jump, l, p, w);
                        Debug.WriteLine($"Epoch: {epoch}, Layer: {l}, Perceptron: {p}, Weight: {w}, NewWeight: {newWeights[l][p][w]}");
                    }
                }

                Weights = newWeights;
            }
        }

        public void Learn(List<double[]> inputs, List<double[]> outputs, int epoch, double omicron = 0.0000001, double alpha = 0.001, int jump = 100)
        {
            if (inputs.Count != outputs.Count)
            {
                throw new ArgumentException("inputs count and outputs count are not samed");
            }

            for (int i = 0; i < epoch; i++)
            {
                for (int j = 0; j < inputs.Count; j++)
                {
                    Epoch(i, inputs[j], outputs[j], omicron, alpha, jump);
                }
            }
        }

        public void Test(double[] input, out double[] output)
        {
            Run(Weights, input, out output);
        }

        public void Save(string filename)
        {
            using(StreamWriter sw = new StreamWriter(filename))
            {
                string json = JsonSerializer.Serialize(Weights);
    
                sw.Write(json);
            }
        }

        public void Load(string filename)
        {
            using (StreamReader sr = new StreamReader(filename))
            {
                string json = sr.ReadToEnd();

                Weights = JsonSerializer.Deserialize<List<List<List<double>>>>(json)!;
            }
        }
    }
}
