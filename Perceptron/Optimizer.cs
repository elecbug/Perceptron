namespace Perceptron
{
    public partial class Classification
    {
        /// <summary>
        /// 사용할 최적화 함수
        /// </summary>
        public enum Optimizer
        {
            /// <summary>
            /// Gradient Descent
            /// </summary>
            GradientDescent,

            /// <summary>
            /// Adam
            /// </summary>
            Adam,
        }
    }
}
