import os


class Config:
    BASE_DIR = "/home/lukas/OneDrive/Dokumente/Uni/ML/Exercise 1/Exercise 1/census"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    PLOTS_DIR = os.path.join(BASE_DIR, "plots")
