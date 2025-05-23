import pandas as pd
from sentence_transformers import SentenceTransformer
import utils
from tqdm import tqdm

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#########
# VAGAS #
#########

df_vagas = pd.read_json(f"{utils.RAW_FILEPATH}/vagas.json", orient="index")
df_vagas = df_vagas.reset_index()
df_vagas = df_vagas.rename(columns={"index": "id_vaga"})
df_info_basicas = pd.json_normalize(df_vagas["informacoes_basicas"])
df_perfil_vaga = pd.json_normalize(df_vagas["perfil_vaga"])
df_beneficios = pd.json_normalize(df_vagas["beneficios"])

df_vagas = pd.concat(
    [df_vagas["id_vaga"].astype(str), df_info_basicas, df_perfil_vaga, df_beneficios],
    axis=1,
)

df_vagas["descricao_vaga"] = (
    df_vagas["titulo_vaga"]
    + "\n"
    + df_vagas["principais_atividades"]
    + "\n"
    + df_vagas["competencia_tecnicas_e_comportamentais"]
)
tqdm.pandas(desc="Traduzindo")

df_vagas["descricao_traduzida"] = df_vagas["descricao_vaga"].progress_apply(
    lambda a: utils.translate(a)
)

print("Criando embeddings das vagas")
df_vagas["embedding"] = model.encode(
    df_vagas["descricao_traduzida"].tolist(), convert_to_numpy=True
).tolist()

df_vagas = df_vagas[
    [
        "id_vaga",
        "titulo_vaga",
        "principais_atividades",
        "competencia_tecnicas_e_comportamentais",
        "descricao_traduzida",
        "embedding",
    ]
]

df_vagas.to_parquet(f"{utils.PROCESSED_FILEPATH}/vagas.parquet", index=False)


#############
# PROSPECTS #
#############

df_prospects = pd.read_json(f"{utils.RAW_FILEPATH}/prospects.json", orient="index")
df_prospects["id_vaga"] = df_prospects.index.astype(str)

df_prospects = df_prospects.explode("prospects").reset_index(drop=True)

prospects = pd.json_normalize(df_prospects["prospects"])
df_prospects = pd.concat(
    [df_prospects.drop(columns=["prospects"]).reset_index(drop=True), prospects], axis=1
)
df_prospects = df_prospects.rename(columns={"codigo": "id_candidato"})

df_prospects = df_prospects[["id_vaga", "id_candidato"]]

df_prospects.to_parquet(f"{utils.PROCESSED_FILEPATH}/prospects.parquet", index=False)


##############
# APPLICANTS #
##############

df_applicants = pd.read_json(f"{utils.RAW_FILEPATH}/applicants.json", orient="index")
df_applicants = df_applicants.reset_index()
df_applicants = df_applicants.rename(columns={"index": "id_candidato"})
informacoes_pessoais = pd.json_normalize(df_applicants["informacoes_pessoais"])
informacoes_profissionais = pd.json_normalize(
    df_applicants["informacoes_profissionais"]
)
formacao_e_idiomas = pd.json_normalize(df_applicants["formacao_e_idiomas"])
cargo_atual = pd.json_normalize(df_applicants["cargo_atual"])
df_cv = df_applicants[["cv_pt", "cv_en"]]

df_applicants = pd.concat(
    [
        df_applicants["id_candidato"].astype(str),
        informacoes_pessoais,
        informacoes_profissionais,
        formacao_e_idiomas,
        cargo_atual,
        df_cv,
    ],
    axis=1,
)

print("Criando embeddings dos candidatos")
df_applicants["embedding"] = model.encode(
    df_applicants["cv_pt"].tolist(), convert_to_numpy=True
).tolist()

df_applicants = df_applicants[
    [
        "id_candidato",
        "nome",
        "email",
        "telefone_celular",
        "cv_pt",
        "embedding",
    ]
]
utils.save_in_chunks(
    df=df_applicants,
    name_function=lambda x: f"{utils.PROCESSED_FILEPATH}/applicants-{x}.parquet",
    chunk_size=5000,
)
