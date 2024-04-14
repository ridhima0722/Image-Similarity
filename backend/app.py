from flask import Flask, jsonify, request,render_template,send_from_directory
import numpy as np
import os
import pickle
from PIL import Image
import io
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
# from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from flask import send_file
from flask_cors import CORS
from skimage import restoration
from skimage.restoration import denoise_wavelet
import cv2


from skimage.restoration import denoise_tv_chambolle
import shutil
from sklearn.cluster import DBSCAN
# from keras.preprocessing import image
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as shc
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import cdist
from sklearn.cluster import MeanShift
# from keras.applications.vgg19 import VGG19, preprocess_input
# from keras.applications.resnet50 import ResNet50
# from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50, preprocess_input
from flask_pymongo import PyMongo



app = Flask(__name__)
CORS(app) 


app.config['MONGO_URI']="mongodb://localhost:27017/image_db"
# app.config['MONGO_URI']="mongodb+srv://ridhima:U9BLGEe9G2I5ftQC@cluster0.1pnjbe9.mongodb.net/image_db"
# app.config['MONGO_URI']="mongodb+srv://ridhima:U9BLGEe9G2I5ftQC@cluster0.1pnjbe9.mongodb.net/image_upload"

mongo=PyMongo(app)

# vgg16 = VGG16(weights='imagenet', include_top=False, pooling='max', input_shape=(224, 224, 3))
resnet50=ResNet50(weights='imagenet',include_top=False,pooling ='max',input_shape=(224,224,3))
# inceptionv3 = InceptionV3(input_shape = (150, 150, 3), include_top = False, weights = 'imagenet')
# vgg19 = VGG19(weights='imagenet', include_top=False, pooling='max', input_shape=(224, 224, 3))

# for model_layer in vgg16.layers:
#     model_layer.trainable = False
# for model_layer in vgg19.layers:
#     model_layer.trainable = False
# for model_layer in resnet50.layers:
#     model_layer.trainable = False
# for model_layer in inceptionv3.layers:
#     model_layer.trainable = False
# resnet50 = Sequential()
# resnet50.add(ResNet50(include_top=False, weights='imagenet', pooling='max'))
# resnet50.add(Dense(1, activation='sigmoid'))
dataset ='products'
image_directory = os.path.join(os.getcwd(), dataset)
image_files = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.endswith(('.png', '.jpeg', 'jpg'))]

if not image_files:
    raise ValueError("no images found")

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
image_generator = datagen.flow_from_directory(image_directory,target_size=(224, 224),batch_size=500,class_mode=None,shuffle=False)

# feature_file_path = 'features.pkl'
feature_file_path = 'resnet50_features.pkl'


if os.path.exists(feature_file_path):
    with open(feature_file_path, 'rb') as file:
        features, image_paths = pickle.load(file)
else:
    features = []
    image_paths = []
    save_batch_size = 100  
    for i, img_path in enumerate(image_files):
        try: 
            img = Image.open(img_path)
            img = img.resize((224, 224))
            # img = image.load_img(img_path, target_size=(224, 224))
        
            # Convert RGBA images to RGB
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            # feature = vgg16.predict(img_array)[0]
            # feature = vgg19.predict(img_array)[0]

            feature = resnet50.predict(img_array)[0]
            # feature = inceptionv3.predict(img_array)


            features.append(feature)
            image_paths.append(img_path)
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")

        if (i + 1) % save_batch_size == 0:
            with open(feature_file_path, 'wb') as file:
                pickle.dump((np.array(features), image_paths), file)

    with open(feature_file_path, 'wb') as file:
        pickle.dump((np.array(features), image_paths), file)

feature_dim = len(features[0])

#this is for resnet50
reshaped_features = features.reshape(-1, 2048)
print("this is the shape of the feature array",reshaped_features.shape)
# print("this is the shape of the feature array",features.shape)
# scaling 
scaler=StandardScaler()
# scaled_feature=scaler.fit_transform(features)
scaled_feature=scaler.fit_transform(reshaped_features)


# After obtaining features using VGG16
# pca = PCA(n_components=300)  
pca=PCA(.95) # retain 95 % of useful featuress 
# reduced_features = pca.fit_transform(features)
reduced_features = pca.fit_transform(scaled_feature)

# Perform hierarchical clustering
# linkage_matrix = linkage(reduced_features, method='average', metric='cosine')
# num_clusters = 18000  # Adjust based on your dataset
# cluster_labels = fcluster(linkage_matrix, t=num_clusters, criterion='maxclust')

# perform mean shift 
mean_shift =MeanShift(bandwidth=0.1)
cluster_labels=mean_shift.fit_predict(reduced_features)
num_clusters=len(np.unique(cluster_labels))
print("this is the umber if clusters",num_clusters)

# denogram ka istemal number of clusters dekhne ke liye but yh large dataset per kaam ni karega 
# plt.figure(figsize=(15, 8))
# dendrogram(linkage_matrix, truncate_mode='level', p=10)
# plt.title('Hierarchical Clustering Dendrogram')
# plt.xlabel('Sample Index or Cluster Size')
# plt.ylabel('Distance')
# plt.show()

# Create a dictionary to store image paths for each cluster
clustered_image_paths = {cluster_id: [] for cluster_id in range( num_clusters)}
for i, image_path in enumerate(image_paths):
    cluster_id = cluster_labels[i]
    clustered_image_paths[cluster_id].append(image_path)

# Create a dictionary to store features for each cluster
clustered_features = {cluster_id: [] for cluster_id in range(num_clusters)}
for i, feature in enumerate(features):
    cluster_id = cluster_labels[i]
    clustered_features[cluster_id].append(feature)

# Convert lists to numpy arrays
for cluster_id in range(num_clusters):
    clustered_features[cluster_id] = np.array(clustered_features[cluster_id])

@app.route('/query_img/', methods=["POST"])
def query_img():
    try:
       
        if request.method == 'POST':
            file = request.files['file']
            image_content = file.read()

            if is_valid_image(file.filename):
                query_feature = extract_features_from_memory(image_content)
                cluster_similarities = {
                    cluster_id: np.max(cosine_similarity_matrix(clustered_features[cluster_id], query_feature))
                    for cluster_id in clustered_features.keys()
                }

                # Choose the top 6 clusters based on similarity with the query image
                top_clusters = sorted(cluster_similarities.items(), key=lambda x: x[1], reverse=True)[:10]

                # Retrieve a single representative image from each of the top clusters
                results = []
                for cluster_id, similarity in top_clusters:
                    cluster_images = clustered_image_paths[cluster_id]
                    if cluster_images:
                        # Choose the first image in the cluster as the representative image
                        representative_image = cluster_images[0]
                        # 22 feb
                        fetch_id=extract_variant_id(representative_image)
                        # print("this is variant id", fetch_id)
                        def_image=mongo.db.variants.find_one({"variantID":int(fetch_id)})
                        x, y = list(def_image.items())[3]
                        path=y
                        results.append({"imgPath": path, "similarity": float(similarity)})

                return jsonify({"results": results})

            else:
                return jsonify({"error": "Select a valid image file (jpg or jpeg)"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    #             # Retrieve a single representative image from each of the top clusters
    #             top_images = []
    #             for cluster_id, similarity in top_clusters:
    #                 cluster_images = clustered_image_paths[cluster_id]
    #                 if cluster_images:
    #                     # Choose the first image in the cluster as the representative image
    #                     representative_image = cluster_images[0]
    #                     top_images.append(representative_image)

    #             results = [{"imgPath": img_path} for img_path in top_images]

    #             return jsonify({"results": results})

    #         else:
    #             return jsonify({"error": "Select a valid image file (jpg or jpeg)"}), 400

    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500
    
                  # Get the cluster label for the query feature
    #             cluster_id = get_cluster_for_query(query_feature, clustered_features)
    #             # Perform similarity search within the cluster
    #             similarities = cosine_similarity_matrix(clustered_features[cluster_id], query_feature)
    #             top_indices = np.argsort(similarities)[::-1][:6]
    #             # similarities = cosine_similarity_matrix(features, query_feature)
    #             # top_indices = np.argsort(similarities)[::-1][:6]
    #             results = []
                 
    #             for index in top_indices:
    #                 result = {
    #                     "similarity": float(similarities[index]),
    #                     # "imgPath": image_files[index]
    #                     "imgPath": clustered_image_paths[cluster_id][index]
    #                 }
    #                 results.append(result)

    #             return jsonify({"results": results})

    #         else:
    #             return jsonify({"error": "Select a valid image file (jpg or jpeg)"}), 400
       
    # except Exception as e:
  
    #     return jsonify({"error": str(e)}), 500
    
def extract_variant_id(image_path):
    variant_id = image_path.split("\\")[-1].split(".")[0]
    return variant_id
def get_cluster_for_query(query_feature, clustered_features):
    # Find the cluster with the highest cosine similarity to the query feature
    cluster_similarities = {
        cluster_id: np.max(cosine_similarity_matrix(clustered_features[cluster_id], query_feature))
        for cluster_id in clustered_features.keys()
    }
    return max(cluster_similarities, key=cluster_similarities.get)

def extract_features_from_memory(image_content):
    img = Image.open(io.BytesIO(image_content))
    img = img.resize((224, 224))
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    # features = vgg19.predict(img_array)
    # features = vgg16.predict(img_array)

    features = resnet50.predict(img_array)
    return features.flatten()


def cosine_similarity_matrix(features, query_feature):
    features_2d = np.array(features).reshape(len(features), -1)
    query_feature_2d = query_feature.reshape(1, -1)
    similarities = cosine_similarity(features_2d, query_feature_2d)
    return similarities.flatten()


def is_valid_image(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.png'))

@app.route('/images/<path:filename>')
def get_image(filename):
    return send_from_directory('products', filename)


if __name__ == "__main__":
    assert os.path.exists('.env')
    os.environ['FLASK_ENV'] = 'development'
    app.run(debug=True)
