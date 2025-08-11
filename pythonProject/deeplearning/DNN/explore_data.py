import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# 显示所有行
pd.set_option('display.max_rows', None)
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示每列的完整内容（不截断字符串）
pd.set_option('display.max_colwidth', None)
# 显示宽度不自动换行
pd.set_option('display.width', None)
def explore_csv(file_path):
    # 1. 读取 CSV 文件
    df = pd.read_csv(file_path)

    print("="*50)
    print("1. 数据基本信息")
    print(f"数据行数: {df.shape[0]}, 列数: {df.shape[1]}")
    print("\n列名:")
    print(df.columns.tolist())
    print("\n数据类型:")
    print(df.dtypes)

    print("="*50)
    print("2. 前5行数据示例:")
    print(df.head())

    print("="*50)
    print("3. 缺失值统计:")
    print(df.isnull().sum())

    print("="*50)
    print("4. 数值列描述性统计:")
    print(df.describe())

    print("="*50)
    print("5. 类别列的分布情况:")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        print(f"\n列名: {col}")
        print(df[col].value_counts())

    print("="*50)
    print("6. 数值特征相关性:")
    numeric_cols = df.select_dtypes(include=['number'])
    if not numeric_cols.empty:
        corr_matrix = numeric_cols.corr()
        print(corr_matrix)

        # # 热力图
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        # plt.title("数值型特征相关性热力图", fontsize=16)
        # plt.show()
    else:
        print("没有数值型特征可计算相关性。")

if __name__ == "__main__":
    file_path = "/Users/mac/PycharmProjects/python_code/pythonProject/deeplearning/DNN/ml2021spring-hw1/covid.train.csv"  # 替换成你的CSV文件路径
    explore_csv(file_path)
