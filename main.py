import pandas as pd
from clean_data import clean_data


if __name__ == "__main__":
    #clean_data("sampled_data.csv", "cleaned_data.csv")

    df = pd.read_csv("cleaned_data.csv")
    category_counts = df['category'].value_counts()

    #pretraining on
    # https://gghantiwala.medium.com/understanding-the-architecture-of-the-inception-network-and-applying-it-to-a-real-world-dataset-169874795540
    # https://medium.com/@sharma.tanish096/detailed-explanation-of-residual-network-resnet50-cnn-model-106e0ab9fa9e
    # https://medium.com/@siddheshb008/vgg-net-architecture-explained-71179310050f
    


    # Print the result
    print(category_counts)