import streamlit as st
import streamlit_vertical_slider as svs
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import sys

# 사용자 정의 simglucose 경로 설정
sys.path.insert(0, './simglucose')

from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.controller.base import Action

# 세션 상태 초기화
if "step" not in st.session_state:
    st.session_state.step = 0
if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = None

st.title("🩺 인슐린 제어 시뮬레이터")

# STEP 0: 환자 선택
if st.session_state.step == 0:
    st.subheader("1️⃣ 환자 선택")
    patient_name = st.selectbox("시뮬레이션할 환자를 선택하세요:", [
            "adult#001", "adult#002", "adult#003", "adult#004", "adult#005", 
            "adult#006", "adult#007", "adult#008", "adult#009", "adult#010",
            "adolescent#001", "adolescent#002", "adolescent#003", "adolescent#004", "adolescent#005", 
            "adolescent#006", "adolescent#007", "adolescent#008", "adolescent#009", "adolescent#010",         
    ])
    csv_file = f"{patient_name}_100_500.csv"

    st.markdown("""
    ### 👨‍⚕️ 인슐린 제어 시뮬레이터 소개

    이 시뮬레이터는 가상의 제1형 당뇨병 환자 데이터를 기반으로,  
    **사용자가 직접 인슐린 주입량을 설정**하고,  
    AI가 제어했을 때의 결과와 비교해볼 수 있는 학습형 플랫폼입니다.

    ---

    #### 🔍 시뮬레이션의 목적
    - 혈당 조절에 있어 인슐린 주입 타이밍과 용량의 중요성을 체험합니다.
    - AI 제어와 비교하여 사용자의 전략이 혈당 안정성에 어떤 영향을 미치는지 확인할 수 있습니다.
    - 실제 당뇨병 치료에 사용되는 기저 인슐린(basal)과 식전 볼루스 인슐린(bolus)의 역할을 구분해 이해할 수 있습니다.

    ---
    """)
    with st.expander("ℹ️ 기저 인슐린과 식전 볼루스 인슐린이란?"):
        st.markdown("""
        #### 💉 인슐린의 두 가지 유형

        **1. 기저 인슐린 (Basal Insulin)**  
        - 하루 종일 일정하게 분비되어 공복 혈당을 조절합니다.  
        - 보통 하루 1~2회 또는 인슐린 펌프를 통해 지속적으로 주입됩니다.

        **2. 식전 볼루스 인슐린 (Bolus Insulin)**  
        - 식사 직전 주입하여 식사 후 급격히 상승하는 혈당을 조절합니다.  
        - 탄수화물 섭취량과 혈당 수치에 따라 용량이 달라집니다.

        ---

        #### 🧠 요약 비교

        | 구분 | 기저 인슐린 (Basal) | 식전 볼루스 인슐린 (Bolus) |
        |------|--------------------|-----------------------------|
        | 목적 | 공복 혈당 조절     | 식후 혈당 조절              |
        | 주입 시기 | 하루 1~2회 또는 지속 주입 | 식사 직전               |
        | 작용 시간 | 느리고 길게        | 빠르고 짧게               |
        """, unsafe_allow_html=True)

    st.markdown("""
    #### 📊 TIR(Time in Range)이란?
    - TIR은 혈당이 70~180 mg/dL 범위 내에 있는 시간의 비율을 의미합니다.
    - TIR이 높을수록 혈당이 안정적으로 유지되며, 당뇨병 관리가 잘 되고 있다는 지표로 사용됩니다.
    - 본 시뮬레이터에서는 AI 제어와 사용자 제어의 TIR을 비교하여 제어 전략의 효과를 평가합니다.

    ---

    👉 아래에서 시뮬레이션할 환자를 선택한 후, 다음 단계로 이동해 주세요.
    """)


    if st.button("다음 단계로"):
        st.session_state.selected_patient = patient_name
        st.session_state.csv_file = csv_file
        st.session_state.step += 1
        st.rerun()

if st.session_state.step == 1:

    st.markdown("""
    #### 📊 그래프 해석 안내

    - **혈당(BG)**과 **CGM 센서**의 값을 비교하여 실제 혈당과 센서 측정값 간의 차이를 확인할 수 있습니다.  
    - **인슐린 주입량**은 혈당이 상승하기 전후로 어떻게 조절되었는지를 시각적으로 보여줍니다.  
    - **CHO(탄수화물 섭취량)** 변화는 식사와 관련된 혈당 상승의 원인을 이해하는 데 도움을 줍니다.

    이러한 그래프는 혈당 변화와 인슐린, 식사의 상관관계를 직관적으로 보여주며, 이후 단계에서의 제어 전략 설정에 중요한 기반이 됩니다.
    """)

    # 전체 시간 구간 시각화
    patient_name = st.session_state.selected_patient
    csv_file = st.session_state.csv_file 

    # 데이터 로드
    df = pd.read_csv(f"data/{st.session_state.csv_file}")
    df["Time"] = pd.to_datetime(df["Time"])

    st.subheader("📈 혈당, 인슐린, CHO 시각화")

    # 혈당 & CGM
    fig_bg_cgm = go.Figure()
    fig_bg_cgm.add_trace(go.Scatter(x=df["Time"], y=df["BG"], name="혈당", line=dict(color="blue")))
    fig_bg_cgm.add_trace(go.Scatter(
        x=df["Time"],
        y=df["CGM"],
        name="CGM",
        line=dict(color="green", dash="dot")
    ))
    fig_bg_cgm.update_layout(
        xaxis_title="시간",
        yaxis_title="혈당 (mg/dL)",
        height=250,
        margin=dict(l=10, r=10, t=30, b=30)
    )
    st.plotly_chart(fig_bg_cgm, use_container_width=True)

    # 인슐린
    fig_insulin = go.Figure()
    fig_insulin.add_trace(go.Scatter(x=df["Time"], y=df["insulin"], name="인슐린", line=dict(color="red")))
    fig_insulin.update_layout(
        xaxis_title="시간",
        yaxis_title="인슐린 (U)",
        height=250,
        margin=dict(l=10, r=10, t=30, b=30)
    )
    st.plotly_chart(fig_insulin, use_container_width=True)
    st.caption("🔴 인슐린 주입량의 시간에 따른 변화")

    # CHO
    fig_cho = go.Figure()
    fig_cho.add_trace(go.Scatter(x=df["Time"], y=df["CHO"], name="CHO", line=dict(color="orange")))
    fig_cho.update_layout(
        xaxis_title="시간",
        yaxis_title="CHO (g)",
        height=250,
        margin=dict(l=10, r=10, t=30, b=30)
    )
    st.plotly_chart(fig_cho, use_container_width=True)
    st.caption("🟠 탄수화물(CHO) 섭취량의 변화")

    # CSV 불러오기 (앱 시작 시 한 번만 실행되게 outside에 둘 수도 있음)
    df_params = pd.read_csv("vpatient_params.csv")  # 경로에 맞게 수정

    # 선택된 환자 이름
    patient_name = st.session_state.selected_patient

    # 해당 환자 데이터 필터링
    patient_info = df_params[df_params["Name"] == patient_name]

    # 정보가 존재할 경우 출력
    if not patient_info.empty:
        info = patient_info.iloc[0]

        st.subheader(f"🧬 `{patient_name}` 환자 요약 정보")

        # 그룹 분류
        if "adolescent" in patient_name:
            group = "청소년"
        elif "adult" in patient_name:
            group = "성인"
        else:
            group = "기타"

        # 주요 생리학적 정보 출력
        st.markdown(f"""
        - **환자 그룹**: {group}  
        - **몸무게 (BW)**: {info['BW']:.3f} kg  
        - **인슐린 감수성 (u2ss)**: {info['u2ss']:.3f}  
        - **간 포도당 생성률 (kp1)**: {info['kp1']:.2f}  
        - **속효성 인슐린 흡수 속도 (ka1)**: {info['ka1']:.4f}  
        - **복부 피하 인슐린 반응 (isc1ss)**: {info['isc1ss']:.2f}  
        """)
    else:
        st.warning("선택한 환자의 정보를 찾을 수 없습니다.")
        
    # STEP 1 내에서 다음 단계 버튼 위에 추가
    st.subheader("🕒 인슐린 제어 시간 구간 선택")
    st.session_state.control_range = st.radio(
        "인슐린을 몇 시간 동안 조절하시겠습니까?",
        ["14시간 (0~14h)", "24시간 (0~24h)"],
        index=0,
        horizontal=True,
        key="control_range_radio"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅️ 이전 단계로") and st.session_state.step > 0:
            st.session_state.step -= 1
            st.rerun()

    with col2:
        if st.button("➡️ 다음 단계로"):
            st.session_df = df
            st.session_state.selected_patient = patient_name
            st.session_state.csv_file = csv_file
            st.session_state.step += 1
            st.rerun()



elif st.session_state.step == 2:
    st.subheader("2️⃣ 사용자 인슐린 제어 설정 (총 14시간)")
    st.caption("각 구간은 시간대에 따라 식사 시점 또는 활동량에 맞춘 인슐린 조절이 필요합니다.")
    st.caption("⏱️ 시간 구간: 0–6h, 6–10h, 10–14h, 기저 인슐린은 전 구간에 적용")
    control_range = st.session_state.control_range  # "14시간 (0~14h)" 또는 "24시간 (0~24h)"

    # 사용자 시뮬레이션 함수 정의
    def simulate_user_response(env_user, dose_bolus, dose_basal):
        bg_user, ins_user, ins_ba = [], [], []
        for bolus in dose_bolus:
            obs, _, _, _ = env_user.step(Action(basal=dose_basal, bolus=bolus))
            bg_user.append(obs[0])
            ins_user.append(bolus)
            ins_ba.append(dose_basal)
        return bg_user, ins_user, ins_ba

    df = st.session_df
    if "14시간" in control_range:
        cols = st.columns(4)
        with cols[0]:
            dose1 = svs.vertical_slider(
            key="dose1",
            default_value=0.03,
            min_value=0,
            max_value=0.05,
            step=0.001,
            slider_color='red',
            track_color='lightgray',
            thumb_color='red',
            height=150,
            value_always_visible=True
            )

            st.caption("0~6h")

        with cols[1]:
            dose2 = svs.vertical_slider(
            key="dose2",
            default_value=0.01,
            min_value=0,
            max_value=0.05,
            step=0.001,
            slider_color='red',
            track_color='lightgray',
            thumb_color='red',
            height=150,
            value_always_visible=True
            )
            st.caption("6~10h")

        with cols[2]:
            dose3 = svs.vertical_slider(
            key="dose3",
            default_value=0.02,
            min_value=0,
            max_value=0.05,
            step=0.001,
            slider_color='red',
            track_color='lightgray',
            thumb_color='red',
            height=150,
            value_always_visible=True
            )
            st.caption("10~14h")

        with cols[3]:
            dose = svs.vertical_slider(
            key="dose",
            default_value=0.02,
            min_value=0,
            max_value=0.05,
            step=0.001,
            slider_color='red',
            track_color='lightgray',
            thumb_color='red',
            height=150,
            value_always_visible=True
            )
            st.caption("기저")

        # 인슐린 주입 시퀀스 구성
        dose_bolus = [dose1]*120 + [dose2]*80 + [dose3]*80
        sim_step = 280

    else:  # 24시간
        cols = st.columns(5)

        with cols[0]:
            dose1 = svs.vertical_slider(
            key="dose1",
            default_value=0.03,
            min_value=0,
            max_value=0.05,
            step=0.001,
            slider_color='red',
            track_color='lightgray',
            thumb_color='red',
            height=150,
            value_always_visible=True
            )

            st.caption("0~6h")

        with cols[1]:
            dose2 = svs.vertical_slider(
            key="dose2",
            default_value=0.01,
            min_value=0,
            max_value=0.05,
            step=0.001,
            slider_color='red',
            track_color='lightgray',
            thumb_color='red',
            height=150,
            value_always_visible=True
            )
            st.caption("6~12h")

        with cols[2]:
            dose3 = svs.vertical_slider(
            key="dose3",
            default_value=0.02,
            min_value=0,
            max_value=0.05,
            step=0.001,
            slider_color='red',
            track_color='lightgray',
            thumb_color='red',
            height=150,
            value_always_visible=True
            )
            st.caption("12~18h")

        with cols[3]:
            dose4 = svs.vertical_slider(
            key="dose4",
            default_value=0.02,
            min_value=0,
            max_value=0.05,
            step=0.001,
            slider_color='red',
            track_color='lightgray',
            thumb_color='red',
            height=150,
            value_always_visible=True
            )
            st.caption("18~24h")

        with cols[4]:
            dose = svs.vertical_slider(
            key="dose",
            default_value=0.02,
            min_value=0,
            max_value=0.05,
            step=0.001,
            slider_color='red',
            track_color='lightgray',
            thumb_color='red',
            height=150,
            value_always_visible=True
            )        
            st.caption("기저인슐린")


        # 인슐린 주입 시퀀스 구성
        dose_bolus = [dose1]*120 + [dose2]*120 + [dose3]*120 +[dose4]*120
        sim_step = 480
    

    st.session_state.doses = dose_bolus

    # 초기 혈당 설정
    ai_df = df.iloc[:sim_step].reset_index(drop=True)
    init_bg = ai_df["BG"].iloc[0]

    # 사용자 시뮬레이션 환경 구성
    sensor = CGMSensor.withName("Dexcom")
    pump = InsulinPump.withName("Insulet")
    patient = T1DPatient.withName(st.session_state.selected_patient, init_state=init_bg)
    scenario = RandomScenario(start_time=datetime.datetime.now(), seed=42)
    env_user = T1DSimEnv(patient=patient, sensor=sensor, pump=pump, scenario=scenario)
    env_user.reset()

    # 사용자 시뮬레이션 실행
    bg_user, ins_user, ins_ba = simulate_user_response(env_user, dose_bolus, dose)

    # AI 데이터 추출
    bg_ai = ai_df["BG"].values
    ins_ai = ai_df["insulin"].values
    if "show_result" not in st.session_state:
        st.session_state.show_result = False

    # 인슐린 그래프
    fig_insulin = go.Figure()
    fig_insulin.add_trace(go.Scatter(x=df["Time"][:sim_step], y=ins_ai, name="AI 인슐린", line=dict(color="orange", dash="dash")))
    fig_insulin.add_trace(go.Scatter(x=df["Time"][:sim_step], y=ins_user, name="사용자 인슐린", line=dict(color="red")))
    fig_insulin.add_trace(go.Scatter(x=df["Time"][:sim_step], y=ins_ba, name="사용자 기저 인슐린", line=dict(color="blue")))
    fig_insulin.update_layout(
        yaxis_title="인슐린 (U)",
        height=300,
        margin=dict(l=10, r=10, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_insulin, use_container_width=True)

    # 결과 혈당 비교
    st.subheader("📈 혈당 비교: AI vs 사용자")
    fig_bg = go.Figure()
    fig_bg.add_trace(go.Scatter(x=df["Time"][:sim_step], y=bg_ai, name="AI 혈당", line=dict(color="orange")))
    fig_bg.add_trace(go.Scatter(x=df["Time"][:sim_step], y=bg_user, name="사용자 혈당", line=dict(color="green")))
    # 정상 혈당 범위 (70~180) 기준선
    fig_bg.add_hline(y=70, line=dict(color="blue", width=1, dash="dot"), name="저혈당 기준")
    fig_bg.add_hline(y=180, line=dict(color="red", width=1, dash="dot"), name="고혈당 기준")

    fig_bg.update_layout(
        yaxis_title="혈당 (mg/dL)",
        height=300,
        margin=dict(l=10, r=10, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_bg, use_container_width=True)

    def analyze_glucose_events(bg_series, time_series):
        """
        저혈당 및 고혈당 발생 구간 요약 반환
        - bg_series: 혈당 값 리스트
        - time_series: 해당 시간에 대응되는 datetime 리스트
        """
        df_g = pd.DataFrame({"time": time_series, "bg": bg_series})
        df_g["status"] = "정상"
        df_g.loc[df_g["bg"] < 70, "status"] = "저혈당"
        df_g.loc[df_g["bg"] > 180, "status"] = "고혈당"

        # 상태 변경 감지
        df_g["group"] = (df_g["status"] != df_g["status"].shift()).cumsum()
        events = df_g[df_g["status"] != "정상"].groupby("group")

        messages = []
        for _, group in events:
            status = group["status"].iloc[0]
            t_start = group["time"].iloc[0].strftime("%H:%M")
            t_end = group["time"].iloc[-1].strftime("%H:%M")
            messages.append(f"- **{t_start} ~ {t_end}** 사이에 **{status}** 발생")

        return messages

    st.markdown("#### 🔍 혈당 결과 해석")

    bg_final = bg_user[-1]
    if bg_final < 70:
        st.warning(f"⚠️ 최종 혈당이 {bg_final:.1f} mg/dL로 저혈당입니다. 인슐린 용량을 줄여보세요.")
    elif bg_final > 180:
        st.warning(f"⚠️ 최종 혈당이 {bg_final:.1f} mg/dL로 고혈당입니다. 인슐린 용량을 늘려볼 수 있습니다.")
    else:
        st.success(f"✅ 최종 혈당 {bg_final:.1f} mg/dL — 안정적인 범위입니다.")

    events = analyze_glucose_events(bg_user, df["Time"][:sim_step])

    st.subheader("🩸 혈당 이상 구간 요약")
    if events:
        for msg in events:
            st.markdown(msg)
    else:
        st.success("✅ 모든 시간대에서 혈당이 정상 범위(70~180 mg/dL)를 유지했습니다.")


    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅️ 이전 단계로") and st.session_state.step > 0:
            st.session_state.step -= 1
            st.rerun()

    with col2:
        if st.button("➡️ 다음 단계로"):
            st.session_state.bg_user = bg_user
            st.session_state.ins_user = ins_user
            st.session_state.bg_ai = bg_ai
            st.session_state.ins_ai = ins_ai
            st.session_state.step += 1
            st.rerun()


# STEP 2: 시뮬레이션 및 결과
elif st.session_state.step == 3:

    # 3. TIR 계산 및 막대 시각화
    def compute_tir(bg_series):
        in_range = np.logical_and(np.array(bg_series) >= 70, np.array(bg_series) <= 180)
        return 100 * np.sum(in_range) / len(bg_series)

    tir_ai = compute_tir(st.session_state.bg_ai)
    tir_user = compute_tir(st.session_state.bg_user)

    fig_tir = go.Figure()
    fig_tir.add_trace(go.Bar(
        x=["AI", "사용자"],
        y=[tir_ai, tir_user],
        marker_color=["green", "blue"]
    ))

    fig_tir.update_layout(
        title="TIR (Time in Range) 비교",
        yaxis_title="TIR (%)",
        xaxis_title="제어 주체",
        yaxis=dict(range=[0, 100]),
        height=400
    )

    st.plotly_chart(fig_tir, use_container_width=True)
    st.subheader("📊 TIR (Time in Range: 70~180 mg/dL)")
    st.write(f"✅ **AI TIR**: {tir_ai:.2f}%")
    st.write(f"🧑‍⚕️ **사용자 TIR**: {tir_user:.2f}%")

    # 결과 비교 메시지
    st.subheader("🏁 결과 요약")
    if tir_user > tir_ai:
        st.success("🎉 **축하합니다!** 사용자 제어가 AI보다 높은 TIR을 기록했습니다!")
    elif tir_user < tir_ai:
        st.error("🤖 아쉽습니다. AI 제어가 더 높은 TIR을 기록했습니다.")
    else:
        st.info("⚖️ 사용자와 AI가 동일한 TIR 성능을 보여주었습니다.")

    def compute_variability(bg_series):
        bg_array = np.array(bg_series)
        avg = np.mean(bg_array)
        std = np.std(bg_array)
        cv = (std / avg) * 100
        return avg, std, cv
    
    # 계산
    avg_ai, std_ai, cv_ai = compute_variability(st.session_state.bg_ai)
    avg_user, std_user, cv_user = compute_variability(st.session_state.bg_user)

    # 표 형태 요약
    st.subheader("📊 혈당 변동성 비교")

    st.markdown(f"""
    | 구분 | 평균 혈당 | 표준편차 (SD) | 변동계수 (CV%) |
    |------|------------|----------------|----------------|
    | **AI** | {avg_ai:.1f} mg/dL | {std_ai:.1f} | {cv_ai:.1f}% |
    | **사용자** | {avg_user:.1f} mg/dL | {std_user:.1f} | {cv_user:.1f}% |
    """, unsafe_allow_html=True)

    # 해석 메시지
    st.markdown("#### 🔍 혈당 변동 해석")
    if cv_user > cv_ai:
        st.warning(f"⚠️ 사용자의 혈당 변동성이 더 큽니다. (CV {cv_user:.1f}% > {cv_ai:.1f}%)")
    else:
        st.success(f"✅ 사용자의 혈당 변동성이 더 낮아 안정적인 패턴을 보였습니다. (CV {cv_user:.1f}% < {cv_ai:.1f}%)")

    # 고위험 경고
    if cv_user > 36:
        st.error("🚨 혈당 변동계수(CV)가 36%를 초과해 고위험군에 해당할 수 있습니다.")

    if st.button("처음으로 돌아가기"):
        st.session_state.step = 0
        st.rerun()