from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


@dataclass
class GeoPoint:
    lat: float
    lon: float

    def to_radians(self) -> "GeoPoint":
        return GeoPoint(np.radians(self.lat), np.radians(self.lon))


class OptimizedGeoDistanceCalculator:
    """벡터화된 배치 처리 방식의 지리적 거리 계산 클래스"""

    EARTH_RADIUS = 6371  # km

    def __init__(self, df_source: pd.DataFrame, df_target: pd.DataFrame):
        self.df_source = df_source
        self.df_target = df_target

        # 타겟 데이터의 좌표를 미리 변환
        self._target_coords = np.radians(df_target[["LATITUDE", "LONGITUDE"]].values)
        self._target_cos_lats = np.cos(self._target_coords[:, 0])

    def _haversine_single(self, lat1: float, lon1: float) -> np.ndarray:
        """
        단일 지점에서의 Haversine 거리 계산

        Parameters:
            lat1: 시작점 위도(라디안)
            lon1: 시작점 경도(라디안)

        Returns:
            모든 타겟 지점까지의 거리 배열
        """
        dlat = self._target_coords[:, 0] - lat1
        dlon = self._target_coords[:, 1] - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * self._target_cos_lats * np.sin(dlon / 2) ** 2

        return 2 * np.arcsin(np.sqrt(a)) * self.EARTH_RADIUS

    def calculate_distances(
        self,
        point: Union[GeoPoint, Tuple[float, float]],
        max_distance: Optional[float] = None,
    ) -> pd.Series:
        """
        주어진 지점으로부터 모든 target 지점까지의 거리 계산

        Parameters:
            point: GeoPoint 객체 또는 (위도, 경도) 튜플
            max_distance: 최대 거리 제한 (km, 선택사항)

        Returns:
            거리가 계산된 Series (인덱스는 target 데이터프레임의 인덱스)
        """
        if isinstance(point, tuple):
            point = GeoPoint(*point)
        point = point.to_radians()

        distances = self._haversine_single(point.lat, point.lon)
        result = pd.Series(distances, index=self.df_target.index)

        if max_distance is not None:
            result = result[result <= max_distance]

        return result

    def find_nearest(self, point: Union[GeoPoint, Tuple[float, float]], k: int = 1) -> pd.DataFrame:
        """
        주어진 지점에서 가장 가까운 k개의 target 지점을 찾음

        Parameters:
            point: GeoPoint 객체 또는 (위도, 경도) 튜플
            k: 반환할 가장 가까운 위치의 개수

        Returns:
            가장 가까운 k개의 위치와 거리가 포함된 DataFrame
        """
        distances = self.calculate_distances(point)
        nearest_idx = distances.nsmallest(k).index

        result = self.df_target.loc[nearest_idx].copy()
        result["distance_km"] = distances[nearest_idx]

        return result

    def _haversine_batch(self, source_coords: np.ndarray) -> np.ndarray:
        """배치 단위 Haversine 거리 계산 (완전 벡터화)"""
        source_lats = source_coords[:, 0:1]
        source_lons = source_coords[:, 1:2]

        dlat = self._target_coords[:, 0] - source_lats
        dlon = self._target_coords[:, 1] - source_lons

        source_cos_lats = np.cos(source_lats)

        a = np.sin(dlat / 2) ** 2 + source_cos_lats * self._target_cos_lats * np.sin(dlon / 2) ** 2

        return 2 * np.arcsin(np.sqrt(a)) * self.EARTH_RADIUS

    def find_nearest_for_all(self, batch_size: int = 1000) -> pd.DataFrame:
        """모든 source 지점에 대해 가장 가까운 target 지점을 찾음"""
        results = []
        total_rows = len(self.df_source)
        total_batches = (total_rows + batch_size - 1) // batch_size

        with tqdm(total=total_batches, desc="Finding nearest points") as pbar:
            for start_idx in range(0, total_rows, batch_size):
                end_idx = min(start_idx + batch_size, total_rows)
                batch_df = self.df_source.iloc[start_idx:end_idx]

                progress = (start_idx + batch_size) / total_rows * 100

                batch_coords = np.radians(batch_df[["LATITUDE", "LONGITUDE"]].values)
                distances = self._haversine_batch(batch_coords)

                nearest_indices = distances.argmin(axis=1)
                min_distances = distances[np.arange(len(distances)), nearest_indices]

                batch_results = pd.DataFrame(
                    {
                        "source_index": batch_df.index,
                        "target_index": self.df_target.index[nearest_indices],
                        "distance_km": min_distances,
                    },
                )

                results.append(batch_results)
                pbar.update(1)
                pbar.set_postfix({"Progress": f"{min(progress, 100):.1f}%"})

        result_df = pd.concat(results, ignore_index=True)

        print("\nMerging results with source and target data...")
        final_df = pd.merge(self.df_source, result_df, left_index=True, right_on="source_index")

        final_df = pd.merge(
            final_df,
            self.df_target,
            left_on="target_index",
            right_index=True,
            suffixes=("_source", "_target"),
        )

        return final_df
