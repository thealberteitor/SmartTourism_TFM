from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
from sklearn.neighbors import NearestNeighbors


class KMedoidsSMOTE:
    def __init__(self, n_clusters=1, sampling_strategy="not majority", metric="manhattan", smote_rate=1, random_state=17):
        self.n_clusters = n_clusters
        self.smote_rate = smote_rate
        self.label_encoder = LabelEncoder()
        self.logic = sampling_strategy
        self.metric = metric
        self.random_state = random_state

    def fit(self, X, y=None):
        return self


    def _generate_synthetic_samples(self, samples, N):
        self._set_random_state(self.random_state)

        n_samples, n_features = samples.shape
        synthetic_samples = np.zeros((N, n_features))

        neighbors = NearestNeighbors(n_neighbors=5, metric=self.metric).fit(samples)
        distances, indices = neighbors.kneighbors(samples)

        for i in range(N):
            idx1 = np.random.randint(0, n_samples)
            nn_idx = np.random.choice(indices[idx1][1:])
            sample1 = samples.iloc[idx1]
            sample2 = samples.iloc[nn_idx]

            diff = sample2 - sample1
            gap = np.random.rand()
            synthetic_samples[i, :] = sample1 + gap * diff

        return pd.DataFrame(synthetic_samples, columns=samples.columns)


    def plot_class_distribution(self, y_before, y_after, title_before='Before Resampling', title_after='After Resampling'):
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        #Before resampling
        y_before.value_counts().plot(kind='bar', ax=axs[0], color='skyblue')
        axs[0].set_title(title_before)
        axs[0].set_xlabel('Class')
        axs[0].set_ylabel('Number of samples')

        #After
        y_after.value_counts().plot(kind='bar', ax=axs[1], color='lightgreen')
        axs[1].set_title(title_after)
        axs[1].set_xlabel('Class')
        axs[1].set_ylabel('Number of samples')

        plt.tight_layout()
        plt.show()

    def fit_resample(self, X, y):
        X_resampled, y_resampled = None, None

        original_X = X.copy()
        original_y = y.copy()

        if self.logic == "all":
            X_resampled, y_resampled = self.fit_resample_all(X, y)
        elif self.logic == "not minority":
            X_resampled, y_resampled = self.fit_resample_not_minority(X, y)
        elif self.logic == "not majority":
            X_resampled, y_resampled = self.fit_resample_not_majority(X, y)
        elif self.logic == "minority":
            X_resampled, y_resampled = self.fit_resample_minority(X, y)
        else:
            print("Error, not implemented")

        assert np.array_equal(original_y, y)
        assert np.array_equal(original_X, X)

        """
        #debug prints
        print(len(original_X))
        print(len(original_y))
        print(len(X_resampled))
        print(len(y_resampled))

        #self.plot_class_distribution(pd.Series(y), pd.Series(y_resampled))
        #self.visualize_clusters_3d(X, y)
        #self.visualize_clusters_3d(X_resampled, y_resampled)
        """
        return X_resampled, y_resampled


    def fit_resample_not_majority(self, X, y):
        """
        #resample all classes but the majority class;
        #clases balanceadas a la clase superior.

        This method is useful when the goal is to preserve the original distribution
        of the majority class while increasing the representation of minority
        classes to create a more balanced dataset.
        """

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        classes = np.unique(y_encoded)

        class_counts = {c: np.sum(y_encoded == c) for c in classes}
        majority_class = max(class_counts, key=class_counts.get)
        target_count = max(class_counts.values())

        X_resampled = pd.DataFrame()
        y_resampled = np.array([])

        for c in classes:
            class_indices = np.where(y_encoded == c)[0]
            class_samples = X.iloc[class_indices, :].copy()

            if c == majority_class:
                X_resampled = pd.concat([X_resampled, class_samples], ignore_index=True)
                y_resampled = np.append(y_resampled, [c] * len(class_samples))
                continue

            if len(class_samples) < self.n_clusters:
                X_resampled = pd.concat([X_resampled, class_samples], ignore_index=True)
                y_resampled = np.append(y_resampled, [c] * len(class_samples))
                continue

            print(self.metric)
            kmedoids = KMedoids(n_clusters=self.n_clusters, metric=self.metric, random_state=self.random_state)
            class_samples['cluster'] = kmedoids.fit_predict(class_samples.drop('cluster', axis=1, errors='ignore'))

            for cluster_label in np.unique(kmedoids.labels_):
                cluster_samples = class_samples[class_samples['cluster'] == cluster_label].drop('cluster', axis=1)

                N = target_count - len(cluster_samples)
                synthetic_samples = self._generate_synthetic_samples(cluster_samples, N)

                class_label = self.label_encoder.inverse_transform([c])[0]
                print(f"{N} synthetic samples generated for class {class_label}")

                X_resampled = pd.concat([X_resampled, cluster_samples, synthetic_samples], ignore_index=True)
                y_resampled = np.append(y_resampled, [c] * (len(cluster_samples) + N))

        y_resampled = self.label_encoder.inverse_transform(y_resampled.astype(int))
        return X_resampled, pd.Series(y_resampled)



    def fit_resample_all(self, X, y):
        """
        Oversample all classes.
        """
        self._set_random_state(self.random_state)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        classes = np.unique(y_encoded)

        X_resampled = pd.DataFrame()
        y_resampled = np.array([])

        for c in classes:
            class_indices = np.where(y_encoded == c)[0]
            class_samples = X.iloc[class_indices, :].copy()

            print(f"Clase {self.label_encoder.inverse_transform([c])[0]}: {len(class_samples)} muestras")

            if len(class_samples) >= self.n_clusters:
                kmedoids = KMedoids(n_clusters=self.n_clusters, metric=self.metric, random_state=self.random_state)
                class_samples['cluster'] = kmedoids.fit_predict(class_samples.drop('cluster', axis=1, errors='ignore'))

                for cluster_label in np.unique(kmedoids.labels_):
                    cluster_samples = class_samples[class_samples['cluster'] == cluster_label].drop('cluster', axis=1, errors='ignore')

                    N = int(len(cluster_samples) * self.smote_rate)
                    synthetic_samples = self._generate_synthetic_samples(cluster_samples, N)

                    print(f"Se generan {N} muestras sintéticas para la clase {self.label_encoder.inverse_transform([c])[0]}")

                    X_resampled = pd.concat([X_resampled, cluster_samples, synthetic_samples], ignore_index=True)
                    y_resampled = np.append(y_resampled, [c] * (len(cluster_samples) + N))
            else:
                X_resampled = pd.concat([X_resampled, class_samples], ignore_index=True)
                y_resampled = np.append(y_resampled, [c] * len(class_samples))

        y_resampled = self.label_encoder.inverse_transform(y_resampled.astype(int))


        return X_resampled, pd.Series(y_resampled)


    def _set_random_state(self, seed=None):
        if seed is not None:
            np.random.seed(seed)


    def visualize_clusters(self, X_resampled, y_resampled):
        self._set_random_state(self.random_state)

        le = LabelEncoder()
        y_resampled_encoded = le.fit_transform(y_resampled)

        #Normalización de los datos
        #scaler = StandardScaler()
        #X_scaled = scaler.fit_transform(X_resampled)

        #scaler = MinMaxScaler(feature_range=(0, 1))
        #X_scaled = scaler.fit_transform(X_resampled)

        #scaler = RobustScaler()
        #X_scaled = scaler.fit_transform(X_resampled)

        #scaler = MaxAbsScaler()
        #X_scaled = scaler.fit_transform(X_resampled)


        scaler = Normalizer()
        X_scaled = scaler.fit_transform(X_resampled)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)  # Usar datos normalizados aquí

        # Preparación del DataFrame para visualización
        df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
        df_pca['Cluster'] = y_resampled_encoded

        # Visualización
        fig, ax = plt.subplots()
        scatter = ax.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['Cluster'], cmap='viridis')

        legend_labels = le.inverse_transform(np.unique(y_resampled_encoded))
        legend_handles, _ = scatter.legend_elements()
        ax.legend(legend_handles, legend_labels, title="Clusters")

        plt.title('Clústeres después de PCA')
        plt.xlabel('Componente 1')
        plt.ylabel('Componente 2')
        plt.show()


    def visualize_clusters_3d(self, X_resampled, y_resampled):
        self._set_random_state(self.random_state)


        le = LabelEncoder()
        y_resampled_encoded = le.fit_transform(y_resampled)

        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_resampled)

        df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])
        df_pca['Cluster'] = y_resampled_encoded

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(df_pca['PC1'], df_pca['PC2'], df_pca['PC3'], c=df_pca['Cluster'], cmap='viridis')
        legend_labels = le.inverse_transform(np.unique(y_resampled_encoded))
        custom_legend = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=5, linestyle='None') for _ in legend_labels]
        ax.legend(custom_legend, legend_labels, title="Clusters")

        plt.title('Clústeres después de PCA')
        ax.set_xlabel('Componente 1')
        ax.set_ylabel('Componente 2')
        ax.set_zlabel('Componente 3')
        plt.show()




    def fit_resample_not_minority(self, X, y):
        """
        Oversample all but minority
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        classes = np.unique(y_encoded)

        class_counts = {c: np.sum(y_encoded == c) for c in classes}
        sorted_counts = sorted(class_counts.values(), reverse=True)
        majority_class = max(class_counts, key=class_counts.get)
        target_count = sorted_counts[1] if len(sorted_counts) > 1 else sorted_counts[0]

        X_resampled = pd.DataFrame()
        y_resampled = np.array([])

        for c in classes:
            class_indices = np.where(y_encoded == c)[0]
            class_samples = X.iloc[class_indices, :].copy()

            if c == majority_class:
                X_resampled = pd.concat([X_resampled, class_samples], ignore_index=True)
                y_resampled = np.append(y_resampled, [c] * len(class_samples))
                continue

            kmedoids = KMedoids(n_clusters=min(self.n_clusters, len(class_samples)), metric=self.metric, random_state=self.random_state)
            class_samples['cluster'] = kmedoids.fit_predict(class_samples.drop(columns=['cluster'], errors='ignore'))

            for cluster_label in np.unique(kmedoids.labels_):
                cluster_samples = class_samples[class_samples['cluster'] == cluster_label].drop(columns=['cluster'], errors='ignore')
                if len(cluster_samples) < target_count:
                    N = target_count - len(cluster_samples)
                    synthetic_samples = self._generate_synthetic_samples(cluster_samples, N)
                    X_resampled = pd.concat([X_resampled, cluster_samples, synthetic_samples], ignore_index=True)
                    print(f"{N} muestras sintéticas generadas para la clase {c}, cluster {cluster_label}.")
                    y_resampled = np.append(y_resampled, [c] * (len(cluster_samples) + N))
                else:
                    print(f"No se necesitan muestras sintéticas para la clase {c}, cluster {cluster_label}.")
                    X_resampled = pd.concat([X_resampled, cluster_samples], ignore_index=True)
                    y_resampled = np.append(y_resampled, [c] * len(cluster_samples))

        y_resampled = self.label_encoder.inverse_transform(y_resampled.astype(int))
        return X_resampled, pd.Series(y_resampled)



    def fit_resample_minority(self, X, y):
        """
        Resample only minority class
        """

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        classes = np.unique(y_encoded)

        class_counts = {c: np.sum(y_encoded == c) for c in classes}
        sorted_counts = sorted(class_counts.values())
        minority_class = min(class_counts, key=class_counts.get)
        target_count = sorted_counts[1] if len(sorted_counts) > 1 else sorted_counts[0]

        X_resampled = pd.DataFrame()
        y_resampled = np.array([])

        for c in classes:
            class_indices = np.where(y_encoded == c)[0]
            class_samples = X.iloc[class_indices, :].copy()

            print(f"Class {self.label_encoder.inverse_transform([c])[0]}: {len(class_samples)} samples")

            if c == minority_class:
                if len(class_samples) < self.n_clusters:
                    X_resampled = pd.concat([X_resampled, class_samples], ignore_index=True)
                    y_resampled = np.append(y_resampled, [c] * len(class_samples))
                else:
                    kmedoids = KMedoids(n_clusters=self.n_clusters, metric=self.metric, random_state=self.random_state)
                    class_samples['cluster'] = kmedoids.fit_predict(class_samples.drop('cluster', axis=1, errors='ignore'))

                    for cluster_label in np.unique(kmedoids.labels_):
                        cluster_samples = class_samples[class_samples['cluster'] == cluster_label].drop('cluster', axis=1, errors='ignore')
                        N = target_count - len(class_samples)  # Updated calculation
                        synthetic_samples = self._generate_synthetic_samples(cluster_samples, N)

                        print(f"{N} synthetic samples generated for the class {self.label_encoder.inverse_transform([c])[0]}")

                        X_resampled = pd.concat([X_resampled, cluster_samples, synthetic_samples], ignore_index=True)
                        y_resampled = np.append(y_resampled, [c] * (len(cluster_samples) + N))
            else:
                X_resampled = pd.concat([X_resampled, class_samples], ignore_index=True)
                y_resampled = np.append(y_resampled, [c] * len(class_samples))

        y_resampled = self.label_encoder.inverse_transform(y_resampled.astype(int))
        return X_resampled, pd.Series(y_resampled)
