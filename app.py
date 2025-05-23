import streamlit as st
import io
import glob
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import utils

st.set_page_config(page_title="FIAP - Tech Challenge 5", layout="centered")

applicant_files = glob.glob(f"{utils.PROCESSED_FILEPATH}/applicants-*.parquet")
applicant_data = [pd.read_parquet(f) for f in applicant_files]
df_applicants = pd.concat(applicant_data, ignore_index=True)
df_prospects = pd.read_parquet(f"{utils.PROCESSED_FILEPATH}/prospects.parquet")
df_vagas = pd.read_parquet(f"{utils.PROCESSED_FILEPATH}/vagas.parquet")

st.header("📋 Recomendador de Candidatos")
st.markdown("---")

with st.form(key="form_busca"):
    col1, col2 = st.columns([3, 2])
    with col1:
        id_vaga = st.text_input("🔎 ID da Vaga")
    with col2:
        quantidade = st.number_input(
            "👥 Qtde. de Currículos", min_value=1, step=1, value=5
        )

    flag_inscrito = st.checkbox("📌 Mostrar apenas candidatos inscritos na vaga")
    search = st.form_submit_button("🔍 Buscar", use_container_width=True)

# Resultado
if search:
    if id_vaga:
        df_vagas_filtrado = df_vagas.loc[df_vagas["id_vaga"] == id_vaga]
        if df_vagas_filtrado.empty:
            st.warning("⚠️ Vaga não encontrada.")
        else:
            df_prospects_vaga = df_prospects.loc[df_prospects["id_vaga"] == id_vaga]
            df_applicants_prospect = df_applicants.merge(
                df_prospects_vaga[["id_candidato"]],
                on="id_candidato",
                how="left",
                indicator=True,
            )
            df_applicants_prospect["inscrito_vaga"] = df_applicants_prospect[
                "_merge"
            ].map({"both": "Sim", "left_only": "Não", "right_only": "Não"})

            df_applicants_prospect = df_applicants_prospect.drop(columns=["_merge"])

            emb_vaga = df_vagas_filtrado["embedding"].values[0]

            if flag_inscrito:
                df_applicants_prospect = df_applicants_prospect.loc[
                    df_applicants_prospect["inscrito_vaga"] == "Sim"
                ]

            sim_scores = cosine_similarity(
                [emb_vaga], np.stack(df_applicants_prospect["embedding"].values)
            ).flatten()

            df_result = df_applicants_prospect.copy()
            df_result["similaridade"] = sim_scores

            df_result = df_result.sort_values("similaridade", ascending=False).head(
                quantidade
            )

            if df_result.empty:
                st.warning("⚠️ Nenhum currículo encontrado.")
            else:
                st.success(f"✅ Top {quantidade} candidatos:")
                for i, row in df_result.iterrows():
                    with st.expander(f"👤 {row['nome']}"):
                        st.markdown(f"📱 **Telefone:** {row['telefone_celular']}")
                        st.markdown(f"📧 **Email:** {row['email']}")
                        st.markdown(f"📌 **Inscrito na vaga:** {row['inscrito_vaga']}")
                        st.text_area("📝 **Currículo:**", row["cv_pt"])

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                df_result.drop(columns=["embedding"]).to_excel(
                    writer,
                    index=False,
                    sheet_name="Candidatos",
                )

            buffer.seek(0)

            st.download_button(
                label="📥 Baixar resultados em Excel",
                data=buffer,
                file_name=f"top_{quantidade}_recomendacoes_vaga_{id_vaga}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
