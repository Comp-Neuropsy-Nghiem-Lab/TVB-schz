import re
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils.tools import expand_path


def extract_label(subject):
    match = re.match(r"[A-Za-z]+", subject)
    return match.group(0) if match else subject


class ClassificationBase:
    def __init__(self, config):
        self.cfg = config
        self.data_dir = expand_path(config["data_dir"])
        self.output_dir = expand_path(config["output_dir"])
        self.parc = config.get("parcellation", 83) or 83
        self.result_exp = config.get("result_exp", "normal")
        self.test_size = config.get("classification_test_size", 0.5)
        self.n_splits = config.get("classification_n_splits", 100)

        self.samples = defaultdict(list)
        self.features_by_metric = {}
        self.fc_features = {}

    def load_results(self):
        for metric in self.cfg["weight_types"]:
            result_dir = self.output_dir / self.result_exp / metric / f"parc{self.parc}"
            if not result_dir.exists():
                continue

            for file_path in sorted(result_dir.glob("*.npy")):
                data = np.load(file_path, allow_pickle=True).item()
                subject = data["subject"]
                self.samples[metric].append(
                    {
                        "subject": subject,
                        "fc_subject": data.get("fc_subject", subject),
                        "label": extract_label(subject),
                        "J_i": np.asarray(data["J_i"]).flatten(),
                        "wLRE": np.asarray(data["wLRE"]).flatten(),
                        "wFFI": np.asarray(data["wFFI"]).flatten(),
                    }
                )

    def build_features(self):
        fc_all = np.load(self.data_dir / self.cfg["fc_file"], allow_pickle=True).item()
        first_metric = next(iter(self.samples), None)

        fc_deg = []
        fc_eig = []
        fc_full = []
        fc_labels = []

        self.features_by_metric = {
            "J_i": {},
            "wLRE": {},
            "wFFI": {},
            "SC_deg": {},
            "SC_eig": {},
            "FC_SC": {},
        }

        for metric, samples in self.samples.items():
            if not samples:
                continue

            sc_all = np.load(
                self.data_dir / self.cfg["weight_types"][metric],
                allow_pickle=True,
            ).item()

            self.features_by_metric["J_i"][metric] = [sample["J_i"] for sample in samples]
            self.features_by_metric["wLRE"][metric] = [sample["wLRE"] for sample in samples]
            self.features_by_metric["wFFI"][metric] = [sample["wFFI"] for sample in samples]

            sc_deg = []
            sc_eig = []
            fc_sc = []
            labels = []

            for sample in samples:
                subject = sample["subject"]
                fc_subject = sample["fc_subject"]

                sc_mat = np.asarray(sc_all[subject][self.parc])
                fc_mat = np.asarray(fc_all[fc_subject][self.parc])

                sc_deg.append(np.sum(sc_mat, axis=1))
                sc_eig.append(np.linalg.eig(sc_mat)[1][:, -1].real)
                fc_sc.append(np.concatenate([fc_mat.flatten(), sc_mat.flatten()]))
                labels.append(sample["label"])

                if metric == first_metric:
                    fc_deg.append(np.sum(fc_mat, axis=1))
                    fc_eig.append(np.linalg.eig(fc_mat)[1][:, -1].real)
                    fc_full.append(fc_mat.flatten())
                    fc_labels.append(sample["label"])

            self.features_by_metric["SC_deg"][metric] = {"X": sc_deg, "y": labels}
            self.features_by_metric["SC_eig"][metric] = {"X": sc_eig, "y": labels}
            self.features_by_metric["FC_SC"][metric] = {"X": fc_sc, "y": labels}
            self.features_by_metric["J_i"][metric] = {"X": self.features_by_metric["J_i"][metric], "y": labels}
            self.features_by_metric["wLRE"][metric] = {"X": self.features_by_metric["wLRE"][metric], "y": labels}
            self.features_by_metric["wFFI"][metric] = {"X": self.features_by_metric["wFFI"][metric], "y": labels}

        self.fc_features = {
            "FC_deg": {"X": fc_deg, "y": fc_labels},
            "FC_eig": {"X": fc_eig, "y": fc_labels},
            "FC_full": {"X": fc_full, "y": fc_labels},
        }

    def build_model(self):
        raise NotImplementedError

    def evaluate(self, X, y):
        scores = []
        X = np.asarray(X)
        y = np.asarray(y)

        for split_id in range(self.n_splits):
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=split_id,
            )
            model = self.build_model()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))

        return {
            "scores": np.asarray(scores),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        }

    def run(self):
        self.load_results()
        self.build_features()

        results = {
            "classifier": self.__class__.__name__,
            "result_exp": self.result_exp,
            "parcellation": self.parc,
            "by_metric": {},
            "fc_only": {},
        }

        for feature_name, feature_dict in self.features_by_metric.items():
            results["by_metric"][feature_name] = {}
            for metric, data in feature_dict.items():
                results["by_metric"][feature_name][metric] = self.evaluate(data["X"], data["y"])

        for feature_name, data in self.fc_features.items():
            if len(data["X"]) > 0:
                results["fc_only"][feature_name] = self.evaluate(data["X"], data["y"])

        return results


class SGD(ClassificationBase):
    def build_model(self):
        return make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss=self.cfg.get("sgd_loss", "log_loss"),
                penalty="l2",
                max_iter=self.cfg.get("sgd_max_iter", 5000),
                tol=self.cfg.get("sgd_tol", 5e-3),
                random_state=self.cfg.get("random_seed", 42),
            ),
        )


class RF(ClassificationBase):
    def build_model(self):
        return RandomForestClassifier(
            n_estimators=self.cfg.get("rf_n_estimators", 300),
            max_depth=self.cfg.get("rf_max_depth", None),
            random_state=self.cfg.get("random_seed", 42),
        )
