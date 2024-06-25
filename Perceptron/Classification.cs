using System.Diagnostics;
using System.Reflection.Emit;
using System.Text.Json;
using Perceptron.Enumerable;
using Perceptron.Internal;

namespace Perceptron
{
    /// <summary>
    /// 분류 작업을 위해 설계된 Deep-learning Core
    /// </summary>
    public class Classification
    {
        /// <summary>
        /// 가중치, Weights[Layer][Perceptron Number of Layer][Weight Number of Perceoptorn]의 구조
        /// </summary>
        private List<List<List<double>>> Weights { get; set; } = new List<List<List<double>>>();

        /// <summary>
        /// 입력 배열의 길이
        /// </summary>
        public int InputCount { get; private set; }

        /// <summary>
        /// 출력 배열의 길이
        /// </summary>
        public int OutputCount { get; private set; }

        /// <summary>
        /// Layer 별 활성화 함수
        /// </summary>
        public List<ActFunc> ActivationFunctions { get; private set; }

        /// <summary>
        /// Model을 생성하고 초기화
        /// </summary>
        /// <param name="inputCount"> 입력 배열의 길이 </param>
        /// <param name="layer"> Layer의 구조, layer.Length는 전체 Layer의 수, layer[Index]는 각 Layer의 Perceptron의 수 </param>
        /// <param name="activationFunctions"> 활성화 함수의 List </param>
        /// <param name="seed"> 시드 값, 기본 값은 0 </param>
        /// <exception cref="ArgumentException"> Layer의 숫자와 활성화 함수의 숫자가 일치하지 않을 시 발생 </exception>
        public Classification(int inputCount, int[] layer, List<ActFunc> activationFunctions, int seed = 0)
        {
            InputCount = inputCount;
            OutputCount = layer.Last();
            ActivationFunctions = activationFunctions;

            if (activationFunctions.Count != layer.Length)
            {
                throw new ArgumentException("Activation function's count and layer's count are not samed");
            }

            // 전역 변수 Weights의 구조를 설정
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

            // 무작위 값을 대입 후 Debug에 출력
            // 후에 Seed를 사용할 수 있게 수정 필요
            for (int l = 0; l < Weights.Count; l++)
            {
                Debug.WriteLine($"L: {l}");

                for (int p = 0; p < Weights[l].Count; p++)
                {
                    Debug.WriteLine($"  P: {p}");

                    for (int w = 0; w < Weights[l][p].Count; w++)
                    {
                        Weights[l][p][w] = new Random(seed++).NextDouble();

                        Debug.WriteLine($"    W: {w}");
                        Debug.WriteLine($"    {Weights[l][p][w]} ");
                    }
                }
            }
        }

        /// <summary>
        /// Load 함수를 통해 호출되는 생성자
        /// </summary>
        /// <param name="structure"></param>
        private Classification(FileStructure structure) 
        {
            Weights = structure.Weights;
            InputCount = structure.InputCount;
            OutputCount = structure.OutputCount;
            ActivationFunctions = structure.ActivationFunctions; 
            
            if (ActivationFunctions.Count != Weights.Count)
            {
                throw new ArgumentException("Activation function's count and layer's count are not samed");
            }
        }

        /// <summary>
        /// 한 번의 실행 결과에 대해, 하나의 입력에서 나오는 출력 결과를 반환
        /// </summary>
        /// <param name="copied"> 전역 가중치를 복사한 매개 변수 </param>
        /// <param name="input"> 하나의 입력 </param>
        /// <param name="output"> 현재 가중치 상태로서 입력에 대해 출력되는 결과 </param>
        /// <exception cref="ArgumentException"> 입력의 수가 초기에 설정한 것과 다를 시 발생 </exception>
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
                    for (int w = 0; w < copied[l][p].Count - 1; w++)
                    {
                        next[p] += copied[l][p][w] * before[w];
                    }

                    next[p] += copied[l][p].Last();
                }

                before = ActFuncImpl.GetActivation(ActivationFunctions[l])(next);
            }

            output = before;
        }

        /// <summary>
        /// Gradient Descent 최적화 함수
        /// * 현재는 Mini Batch가 지원되지 않아, 사용하는 쪽에서 입력을 묶어서 임의로 Mini Batch를 만들어 사용해야 함
        /// </summary>
        /// <param name="inputs"> 최적화 할 모든 입력 </param>
        /// <param name="outputs"> 최적화 할 모든 출력 </param>
        /// <param name="omicron"> 미분소, 미분을 위한 분모 </param>
        /// <param name="alpha"> 학습률 </param>
        /// <param name="jump"> 회당 학습 수, 학습률의 역수를 추천 </param>
        /// <param name="l"> GD가 진행되는 Layer </param>
        /// <param name="p"> GD가 진행되는 Perceptron </param>
        /// <param name="w"> GD가 진행되는 Weight </param>
        /// <returns></returns>
        private double GradientDescent(List<double[]> inputs, List<double[]> outputs, 
            double omicron, double alpha, int jump, int l, int p, int w)
        {
            // 원본에 지장을 안주기 위해 사본 생성
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

            // 특정 가중치에 대해 jump 만큼 최적화
            double weight = copied[l][p][w];

            for (int i = 0; i < jump; i++)
            {
                double gradientSum = 0;

                for (int j = 0; j < inputs.Count; j++)
                {
                    double[] input = inputs[j];
                    double[] output = outputs[j];

                    copied[l][p][w] += omicron;
                    Run(copied, input, out double[] oPlus);

                    copied[l][p][w] -= 2 * omicron;
                    Run(copied, input, out double[] oMinus);

                    copied[l][p][w] += omicron;

                    double errorPlus = 0;
                    double errorMinus = 0;

                    for (int o = 0; o < output.Length; o++)
                    {
                        errorPlus += (output[o] - oPlus[o]) * (output[o] - oPlus[o]);
                        errorMinus += (output[o] - oMinus[o]) * (output[o] - oMinus[o]);
                    }

                    gradientSum += (errorPlus - errorMinus) / (2 * omicron);
                }

                double averageGradient = gradientSum / inputs.Count;
                
                weight -= alpha * averageGradient;
                copied[l][p][w] = weight;
            }

            return weight;
        }

        /// <summary>
        /// Adam 최적화 함수
        /// </summary>
        /// <param name="inputs"> 최적화 할 모든 입력 </param>
        /// <param name="outputs"> 최적화 할 모든 출력 </param>
        /// <param name="omicron"> 미분소, 미분을 위한 분모 </param>
        /// <param name="alpha"> 학습률 </param>
        /// <param name="jump"> 회당 학습 수, 학습률의 역수를 추천 </param>
        /// <param name="l"> Adam이 진행되는 Layer </param>
        /// <param name="p"> Adam이 진행되는 Perceptron </param>
        /// <param name="w"> Adam이 진행되는 Weight </param>
        /// <param name="beta1"> Adam Parameter beta1 </param>
        /// <param name="beta2"> Adam Parameter beta2 </param>
        /// <param name="epsilon"> Adam Parameter epsilon </param>
        /// <returns></returns>
        private double Adam(List<double[]> inputs, List<double[]> outputs,
            double omicron, double alpha, int jump, int l, int p, int w,
            double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        {
            // 원본에 지장을 안주기 위해 사본 생성
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

            double m = 0;
            double v = 0;
            int t = 0;

            for (int i = 0; i < jump; i++)
            {
                double gradientSum = 0;

                for (int j = 0; j < inputs.Count; j++)
                {
                    double[] input = inputs[j];
                    double[] output = outputs[j];

                    copied[l][p][w] += omicron;
                    Run(copied, input, out double[] oPlus);

                    copied[l][p][w] -= 2 * omicron;
                    Run(copied, input, out double[] oMinus);

                    copied[l][p][w] += omicron;

                    double errorPlus = 0;
                    double errorMinus = 0;

                    for (int o = 0; o < output.Length; o++)
                    {
                        errorPlus += (output[o] - oPlus[o]) * (output[o] - oPlus[o]);
                        errorMinus += (output[o] - oMinus[o]) * (output[o] - oMinus[o]);
                    }

                    gradientSum += (errorPlus - errorMinus) / (2 * omicron);
                }

                double averageGradient = gradientSum / inputs.Count;

                // Adam 업데이트
                t++;
                m = beta1 * m + (1 - beta1) * averageGradient;
                v = beta2 * v + (1 - beta2) * (averageGradient * averageGradient);

                double mHat = m / (1 - Math.Pow(beta1, t));
                double vHat = v / (1 - Math.Pow(beta2, t));

                weight -= alpha * mHat / (Math.Sqrt(vHat) + epsilon);
                copied[l][p][w] = weight;
            }

            return weight;
        }

        /// <summary>
        /// 한 Epoch
        /// </summary>
        /// <param name="epoch"> 지금 몇번째 Epoch인지/출력을 위해 사용 </param>
        /// <param name="totalEpoch"> 총 Epoch 수 </param>
        /// <param name="inputs"> 최적화 할 모든 입력 </param>
        /// <param name="outputs"> 최적화 할 모든 출력 </param>
        /// <param name="omicron"> 미분소, 미분을 위한 분모</param>
        /// <param name="alpha"> 학습률 </param>
        /// <param name="jump"> 회당 학습 수, 학습률의 역수를 추천</param>
        /// <param name="logging"> Log를 생성할 위치 </param>
        /// <param name="optimizer"> 사용할 최적화 함수 </param>
        /// <param name="checkTime"> 현재 시간과 예측치를 Log로 출력할 지 여부 </param>
        /// <param name="parallelMode"> 병렬 모드를 사용할 지 여부 </param>
        private void Epoch(int totalEpoch, int epoch, List<double[]> inputs, List<double[]> outputs, 
            double omicron, double alpha, int jump, Logging logging, Optimizer optimizer, bool parallelMode, bool checkTime)
        {
            // 새 가중치들의 임시 저장 공간
            List<List<List<double>>> newWeights = new List<List<List<double>>>();

            // 총 가중치 개수
            int allWeightCount = 0;

            // 파일 쓰기용 락
            object locker = new object();

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

                    allWeightCount += Weights[ll][pp].Count;
                }
            }

            int wCount = 0;
            decimal average = 0;

            var insideFunc = (int l, int p, int w) => 
            {
                decimal time = DateTime.Now.Ticks;

                switch (optimizer)
                {
                    case Optimizer.GradientDescent:
                        newWeights[l][p][w] = GradientDescent(inputs, outputs, omicron, alpha, jump, l, p, w);
                        break;
                    case Optimizer.Adam:
                        newWeights[l][p][w] = Adam(inputs, outputs, omicron, alpha, jump, l, p, w);
                        break;
                }

                time = DateTime.Now.Ticks - time;
                wCount++;

                average *= (wCount - 1) / (decimal)wCount;
                average += time / wCount;

                if (checkTime == false)
                {
                    switch (logging)
                    {
                        case Logging.Debug:
                            Debug.WriteLine($"Epoch: {epoch}, Layer: {l}, Perceptron: {p}, " +
                                $"Weight: {w}, NewWeight: {newWeights[l][p][w]}");
                            break;
                        case Logging.Console:
                            Console.WriteLine($"Epoch: {epoch}, Layer: {l}, Perceptron: {p}, " +
                                $"Weight: {w}, NewWeight: {newWeights[l][p][w]}");
                            break;
                        case Logging.FileStream:
                            using (StreamWriter sw = new StreamWriter("log.log", true))
                            {
                                sw.WriteLine($"Epoch: {epoch}, Layer: {l}, Perceptron: {p}, " +
                                    $"Weight: {w}, NewWeight: {newWeights[l][p][w]}");
                            }
                            break;
                    }
                }
                else
                {
                    DateTime maybe = DateTime.Now.AddTicks((long)((allWeightCount - wCount) * average));

                    for (int e = epoch + 1; e < totalEpoch; e++)
                    {
                        maybe = maybe.AddTicks((long)(allWeightCount * average));
                    }

                    switch (logging)
                    {
                        case Logging.Debug:
                            Debug.WriteLine($"[to {maybe}] Epoch: {epoch}, Layer: {l}, Perceptron: {p}, " +
                                $"Weight: {w}, NewWeight: {newWeights[l][p][w]}");
                            break;
                        case Logging.Console:
                            Console.WriteLine($"[to {maybe}] Epoch: {epoch}, Layer: {l}, Perceptron: {p}, " +
                                $"Weight: {w}, NewWeight: {newWeights[l][p][w]}");
                            break;
                        case Logging.FileStream:
                            lock (locker)
                            {
                                using (StreamWriter sw = new StreamWriter("log.log", true))
                                {
                                    sw.WriteLine($"Epoch: {epoch}, Layer: {l}, Perceptron: {p}, " +
                                        $"Weight: {w}, NewWeight: {newWeights[l][p][w]}");
                                }
                            }
                            break;
                    }
                }
            };

            // 뒤쪽 레이어부터 최적화
            for (int l = Weights.Count - 1; l >= 0; l--)
            {
                for (int p = 0; p < Weights[l].Count; p++)
                {
                    if (parallelMode == true)
                    {
                        Parallel.For(0, Weights[l][p].Count, w =>
                        {
                            insideFunc(l, p, w);
                        });
                    }
                    else
                    {
                        for (int w = 0; w < Weights[l][p].Count; w++)
                        {
                            insideFunc(l, p, w);
                        }
                    }
                }

                Weights = newWeights;
            }
        }

        /// <summary>
        /// 해당 입출력 값으로 모델을 학습
        /// </summary>
        /// <param name="inputs"> 최적화 할 모든 입력</param>
        /// <param name="outputs"> 최적화 할 모든 출력 </param>
        /// <param name="epoch"> 학습 Epoch 수 </param>
        /// <param name="omicron"> 미분소, 미분을 위한 분모, 기본 값은 0.0000001 </param>
        /// <param name="alpha"> 학습률, 기본 값은 0.001 </param>
        /// <param name="jump"> 회당 학습 수, 학습률의 역수를 추천, 기본 값은 1000 </param>
        /// <param name="logging"> Log를 생성할 위치 </param>
        /// <param name="optimizer"> 사용할 최적화 함수 </param>
        /// <param name="autoSave"> 자동 저장 파일, 빈 문자열 일 시 자동 저장 되지 않음 </param>
        /// <param name="checkTime"> 현재 시간과 예측치를 Log로 출력할 지 여부 </param>
        /// <param name="parallelMode"> 병렬 모드를 사용할 지 여부, 기본 값은 true </param>
        /// <exception cref="ArgumentException"> 입력 배열의 전체 갯수와 출력 배열의 전체 갯수가 다를 시 발생 </exception>
        public void Learn(List<double[]> inputs, List<double[]> outputs, int epoch,
            Logging logging, Optimizer optimizer, bool parallelMode = true,
            string autoSave = "", bool checkTime = true,
            double omicron = 0.0000001, double alpha = 0.001, int jump = 1000)
        {
            if (inputs.Count != outputs.Count)
            {
                throw new ArgumentException("inputs count and outputs count are not samed");
            }

            for (int i = 0; i < epoch; i++)
            {
                Epoch(epoch, i, inputs, outputs, omicron, alpha, jump, logging, optimizer, parallelMode, checkTime);

                if (autoSave != "")
                {
                    Save(autoSave);
                }
            }
        }

        /// <summary>
        /// 현재 학습 상태를 사용하여 계산하고 결과를 반환
        /// </summary>
        /// <param name="input"> 사용할 예시 입력 </param>
        /// <param name="output"> 예시 입력에 대한 출력</param>
        public void Test(double[] input, out double[] output)
        {
            Run(Weights, input, out output);
        }

        /// <summary>
        /// 가중치를 Json 형태로 저장
        /// </summary>
        /// <param name="filename"> 저장할 파일 위치 </param>
        public void Save(string filename)
        {
            using(StreamWriter sw = new StreamWriter(filename))
            {
                FileStructure structure = new FileStructure()
                {
                    Weights = Weights,
                    InputCount = InputCount,
                    OutputCount = OutputCount,
                    ActivationFunctions = ActivationFunctions,
                };

                string json = JsonSerializer.Serialize(structure);
    
                sw.Write(json);
            }
        }

        /// <summary>
        /// Json 형태의 가중치를 불러와 세팅
        /// </summary>
        /// <param name="filename"> 불러올 파일 위치 </param>
        public static Classification Load(string filename)
        {
            using (StreamReader sr = new StreamReader(filename))
            {
                string json = sr.ReadToEnd();

                FileStructure structure = JsonSerializer.Deserialize<FileStructure>(json)!;

                Classification result = new Classification(structure);
                return result;
            }
        }

        private class FileStructure
        {
            public required List<List<List<double>>> Weights { get; set; }
            public required int InputCount { get; set; }
            public required int OutputCount { get; set; }
            public required List<ActFunc> ActivationFunctions { get; set; }
        }
    }
}
