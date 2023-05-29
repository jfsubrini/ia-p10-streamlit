# -*- coding: utf-8 -*-
# pylint: disable=no-member,line-too-long
"""
Created by Jean-François Subrini on the 26th of May 2023.
Creation of a dashboard for "Projet 10 : Développez une preuve de concept".
Simple Streamlit website with 4 "pages" to choose from the sidebar:
Bienvenue / Jeu de données / Segmentation sémantique / A propos du model SOTA
"""
import cv2
from matplotlib import pyplot as plt
import streamlit as st
from sem_seg_utils import (
    NAME_LIST,
    IMG_LIST,
    MSK_LIST,
    mask_prediction,
    unet_model,
    HRNEtOCR_model
    )


# Creating the title for all 4 "pages".
st.title(":blue[PROJET 10 - Développez une POC]")

# Deleting the hamburger and the footer of the original Streamlit page.
st.markdown("""
            <style>
            .css-nqowgj.edgvbvh3
            {
                visibility: hidden;
            }
            .css-h5rgaw.egzxvld1
            {
                visibility: hidden;
            }
            </style>
            """, unsafe_allow_html=True)

# Creating the side bar with the logos and the pages to select.
st.sidebar.title(":blue[TABLEAU DE BORD]")
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.image("img_notebook/logo_dataspace.png", width=250)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
page_sel = st.sidebar.radio(
    "Pages", options=("Bienvenue",
                      "Jeu de données",
                      "Segmentation sémantique",
                      "A propos de notre modèle SOTA"))
st.sidebar.markdown("<br><br><br><br>", unsafe_allow_html=True)
st.sidebar.image("img_notebook/streamlit_logo.png", width=150)

##############################################################################

# Page "Segmentation sémantique".
if page_sel == "Segmentation sémantique":
    st.header(":orange[Segmentation sémantique d'une image]")
    st.markdown("<br>", unsafe_allow_html=True)
    form = st.form("Form 1")
    img_selected = form.selectbox("Sélectionner une image : ",
                                  options=(NAME_LIST))
    submit_state = form.form_submit_button("Valider")
    if submit_state:
        st.success(f"Vous avez sélectionné l'image **{img_selected}** pour réaliser une \
            segmentation sémantique avec le modèle U-NET puis le modèle HRNetV2 + OCR.")
        # Displaying the selected image.
        img_name_id = NAME_LIST.index(img_selected)  # index in the list of the selected image name.
        st.image(f"media/images/{IMG_LIST[img_name_id]}",
                 caption=f"Image {img_selected}", width=500)
        # Displaying the selected mask.
        st.image(f"media/masks/{MSK_LIST[img_name_id]}",
                 caption=f"Masque {img_selected}", width=500)
        # Preparing the image for prediction.
        img_to_predict = cv2.imread(f"media/images/{IMG_LIST[img_name_id]}") / 255.0
        # Predicting the mask with the U-NET model.
        pred_mask_colored = mask_prediction(unet_model, img_to_predict)
        # Displaying the predicted mask with the U-NET model.
        st.image(pred_mask_colored, caption=f"Masque prédit {img_selected} avec U-NET", width=500)
        # Predicting the mask with the HRNetV2 + OCR model.
        pred_mask_colored2 = mask_prediction(unet_model, img_to_predict)  # TODO change model
        # Displaying the predicted mask with the HRNetV2 + OCR model.
        st.image(pred_mask_colored2, caption=f"Masque prédit {img_selected} avec HRNetV2 + OCR", width=500)

##############################################################################

# Page "Jeu de données".
elif page_sel == "Jeu de données":
    # Cityscapes' global dataset.
    st.header(":orange[Jeu de données global de Cityscapes]")
    st.markdown("""
                Nous nous basons ici sur le **jeu de données de Cityscapes** du Projet 8.
                """)
    st.image("img_notebook/logo_cityscapes.png", width=200)
    st.markdown("""
                Il se concentre sur la compréhension sémantique des scènes de rue urbaines, 
                avec ses 5 000 images et masques de haute qualité.
                """)
    st.markdown("----")
    # Radio button to select the type of chart to display.
    opt = st.radio("Sélectionner un diagramme", options=(
        "Diagramme à bâtons", "Diagramme à bâtons horizontal", "Diagramme camembert"))
    # Type of design for the charts.
    URL = "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle"
    # Bar plot choice.
    if opt == "Diagramme à bâtons":
        st.markdown("<h3 style='text-align: center;'>Diagramme à bâtons</h3>",
                    unsafe_allow_html=True)
        fig = plt.figure()
        plt.style.use(URL)
        plt.bar(['training', 'validation', 'test'],
                [2975, 500, 1525],
                color=['blue', 'orange', 'green'])
        st.write(fig)
    # Horizontal bar plot choice.
    elif opt == "Diagramme à bâtons horizontal":
        st.markdown("<h3 style='text-align: center;'>Diagramme à bâtons horizontal</h3>",
                    unsafe_allow_html=True)
        fig = plt.figure()
        plt.style.use(URL)
        plt.barh(['training', 'validation', 'test'],
                 [2975, 500, 1525],
                 color=['blue', 'orange', 'green'])
        st.write(fig)
    # Pie chart choice.
    elif opt == "Diagramme camembert":
        st.markdown("<h3 style='text-align: center;'>Diagramme camembert</h3>",
                    unsafe_allow_html=True)
        fig = plt.figure()
        plt.style.use(URL)
        plt.pie([2975, 500, 1525],
                labels=['training', 'validation', 'test'],
                colors=['blue', 'orange', 'green'],
                explode = (0.1, 0, 0),
                autopct='%1.1f%%')
        st.write(fig)

    # Dataset finally used for training and performance evaluation of the model.
    st.markdown("----")
    st.header(":orange[Jeu de données pour training et évaluation]")
    # Radio button to select the type of chart to display.
    opt = st.radio("Sélectionner un diagramme", options=(
        "Diagramme à bâtons", "Diagramme camembert"))
    # Bar plot choice.
    if opt == "Diagramme à bâtons":
        st.markdown("<h3 style='text-align: center;'>Diagramme à bâtons</h3>",
                    unsafe_allow_html=True)
        fig = plt.figure()
        plt.style.use(URL)
        plt.bar(['training', 'validation'], [744, 125], color=['blue', 'orange'])
        st.write(fig)
    # Pie chart choice.
    elif opt == "Diagramme camembert":
        st.markdown("<h3 style='text-align: center;'>Diagramme camembert</h3>",
                    unsafe_allow_html=True)
        fig = plt.figure()
        plt.style.use(URL)
        plt.pie([744, 125],
                labels=['training', 'validation'],
                colors=['blue', 'orange'],
                explode = (0.1, 0),
                autopct='%1.1f%%')
        st.write(fig)

##############################################################################

# Page "A propos de notre modèle SOTA".
elif page_sel == "A propos de notre modèle SOTA":
    st.header(":orange[Références bibliographiques et autres]")
    st.subheader(":orange[Articles de recherche]")
    st.markdown("""
                [**High-Resolution Representations for Labeling Pixels and Regions**](https://arxiv.org/pdf/1904.04514.pdf), 
                publié le 9 avril 2019.<br>
                [**Deep High-Resolution Representation Learning for Visual Recognition**](https://arxiv.org/pdf/1908.07919.pdf), 
                publié le 13 mars 2020.<br>
                [**Segmentation Transformer: Object-Contextual Representations for Semantic Segmentation**](https://arxiv.org/pdf/1909.11065.pdf), publié le 30 avril 2021.
                """, unsafe_allow_html=True)
    st.subheader(":orange[Site web de Jingdong Wang]")
    st.image("img_notebook/Jingdong_Wang.png", width=100)
    st.markdown("""
                [Site web de Jingdong Wang](https://jingdongwang2017.github.io/Projects/HRNet/), 
                Principal Research Manager de Microsoft Research Lab - Asia, à l’origine de ces modèles.
                """)
    st.subheader(":orange[Implémentation PyTorch des différents modèles HRNet]")
    st.markdown("""
                Site GitHub [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation)
                """)
    st.markdown("----")
    st.subheader(":orange[TensorFlow Advanced Segmentation Models]")
    st.markdown("<br>", unsafe_allow_html=True)
    st.image("img_notebook/tasm.png", width=250)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
                Nous avons utilisé la bibliothèque Python TensorFlow 
                [TASM, de Jan-Marcel Kezmann](https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models), 
                « ***A Python Library for High-Level Semantic Segmentation Models*** », 
                pour la création du modèle HRNetV2 + OCR.
                """)
    st.markdown("----")
    st.markdown(
        "***[Repository GitHub du Notebook du Projet 10](https://github.com/jfsubrini/ai-project-10)***")

##############################################################################

# Page "Bienvenue", landing page.
else:
    st.header(":orange[Test d’un nouveau modèle de segmentation sémantique]")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
              Pour cette **preuve de concept (POC)**, nous reprenons le **Projet 8** du parcours de formation 
              « Ingénieur IA » d’OpenClassrooms : **Participez à la conception d'une voiture autonome**.
              """)
    st.markdown("""
              Suite à une veille technologique nous avons choisi un nouveau modèle *state-of-the-art* récent 
              pour **améliorer la performance du modèle de segmentation sémantique** : le modèle **HRNetV2 + OCR**.
              """)
    st.markdown("""
                Rappelons que la **segmentation sémantique** est le **classement de chaque pixel selon la classe de l'objet 
                auquel il appartient** (humain, véhicule, construction, ciel, etc.) et les différents objets
                d'une même classe ne sont pas distingués. Ce travail est utilisé par les véhicules autonomes
                pour **comprendre leur environnement**.
                """)
