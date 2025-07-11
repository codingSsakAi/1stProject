import pandas as pd


def load_data(filepath: str, nationality: str, start_ym: str, end_ym: str) -> pd.DataFrame:
    """
    전처리된 CSV 파일에서 원하는 국적과 기간의 데이터를 불러오는 함수
    Args:
        filepath (str): CSV 파일 경로
        nationality (str): 필터링할 국적명
        start_ym (str): 시작 연월 (예: '2005-01')
        end_ym (str): 종료 연월 (예: '2005-12')
    Returns:
        pd.DataFrame: 필터링된 데이터프레임
    """
    # CSV 파일을 DataFrame으로 읽어옵니다 (한글 지원을 위해 utf-8-sig 인코딩 사용)
    df = pd.read_csv(filepath, encoding="utf-8-sig")

    # 필요한 컬럼이 모두 있는지 확인합니다
    required_cols = {"국적", "연도", "월", "목적", "입국자수", "코로나기간"}
    if not required_cols.issubset(df.columns):
        # 컬럼이 하나라도 없으면 에러 발생
        raise ValueError(f"필요한 컬럼이 없습니다. 실제 컬럼: {df.columns.tolist()}")

    # 연월(YYYY-MM) 컬럼을 새로 만듭니다 (월이 한 자리면 0을 붙여줌)
    df["연월"] = df["연도"].astype(str) + "-" + df["월"].astype(str).str.zfill(2)

    # 국적 기준으로 데이터 필터링
    df = df[df["국적"] == nationality]

    # 연월 기준으로 기간 필터링
    df = df[(df["연월"] >= start_ym) & (df["연월"] <= end_ym)]

    # 필요한 컬럼만 남기고 반환 (반드시 DataFrame 형태로 반환)
    result = df[["국적", "목적", "연월", "입국자수", "코로나기간"]]
    return pd.DataFrame(result)


if __name__ == "__main__":
    # 테스트용 파라미터 설정
    filepath = "../data_preprocessing/data/processed/외국인입국자_전처리완료_딥러닝용.csv"
    nationality = "미국"  # 실제 데이터에 존재하는 국적명
    start_ym = "2005-01"
    end_ym = "2005-12"
    # 데이터 불러오기 함수 실행
    df = load_data(filepath, nationality, start_ym, end_ym)
    # 결과 일부 출력
    print(df.head())
    print(f"총 {len(df)}행")
