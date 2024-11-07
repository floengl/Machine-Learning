import os


class Config:
    BASE_DIR = "/home/lukas/OneDrive/Dokumente/Uni/ML/Machine-Learning/Exercise 1/amazon"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    SUBMISSION_DIR = os.path.join(BASE_DIR, "submissions")
    PLOTS_DIR = os.path.join(BASE_DIR, "plots")
