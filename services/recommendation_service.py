from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from typing import List
from models.schemas import JobRecommendation
import config
import ast

class RecommendationService:
    def __init__(self, model, dataset_path):
        self.model = model
        self.df_all = pd.read_csv(dataset_path)
        self.minat_encoder = MultiLabelBinarizer()
        self.skills_encoder = MultiLabelBinarizer()
        self.kondisi_encoder = MultiLabelBinarizer()
        self._initialize_encoders()

    def _initialize_encoders(self):
        self.df_all['Kategori Minat'] = self.df_all['Kategori Minat'].apply(self._safe_eval)
        self.df_all['Kemampuan Dibutuhkan'] = self.df_all['Kemampuan Dibutuhkan'].apply(self._safe_eval)
        self.df_all['Kondisi Kesehatan'] = self.df_all['Kondisi Kesehatan'].apply(self._safe_eval)
        
        self.minat_encoder.fit(self.df_all['Kategori Minat'])
        self.skills_encoder.fit(self.df_all['Kemampuan Dibutuhkan'])
        self.kondisi_encoder.fit(self.df_all['Kondisi Kesehatan'])

    @staticmethod
    def _safe_eval(x):
        """Safely evaluate string representations of lists"""
        if pd.isna(x):
            return []
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            if isinstance(x, str):
                x = x.strip('[]')
                if ',' in x:
                    return [item.strip().strip("'\"") for item in x.split(',')]
                return [x.strip().strip("'\"")]
            return [str(x)]

    def process_input_features(self, minat_user: List[str], kemampuan_user: List[str], kondisi_user: List[str]):
        try:
            user_minat_vector = self.minat_encoder.transform([minat_user])
            user_skills_vector = self.skills_encoder.transform([kemampuan_user])
            user_conditions_vector = self.kondisi_encoder.transform([kondisi_user])

            combined_vector = np.hstack([
                user_minat_vector * config.MINAT_WEIGHT,
                user_skills_vector * config.SKILLS_WEIGHT,
                user_conditions_vector * config.CONDITIONS_WEIGHT
            ])

            return combined_vector

        except Exception as e:
            raise ValueError(f"Error processing input features: {str(e)}")

    def get_recommendations(self, minat_user: List[str], kemampuan_user: List[str], kondisi_user: List[str], top_n: int = 5):
        try:
            user_vector = self.process_input_features(minat_user, kemampuan_user, kondisi_user)
            user_prediction = self.model.predict(user_vector)

            job_features = np.hstack([
                self.minat_encoder.transform(self.df_all['Kategori Minat'].tolist()),
                self.skills_encoder.transform(self.df_all['Kemampuan Dibutuhkan'].tolist()),
                self.kondisi_encoder.transform(self.df_all['Kondisi Kesehatan'].tolist())
            ])

            similarity_scores = cosine_similarity(user_prediction, job_features)
            similar_indices = similarity_scores[0].argsort()[-top_n:][::-1]

            recommendations = []
            for idx in similar_indices:
                job = self.df_all.iloc[idx]
                recommendations.append(JobRecommendation(
                    Nama_Pekerjaan=job["Nama Pekerjaan"],
                    Perusahaan=job["Perusahaan"]
                ))

            return recommendations

        except Exception as e:
            raise Exception(f"Error generating recommendations: {str(e)}")