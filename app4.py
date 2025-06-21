import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import datetime
import sys
import plotly.graph_objects as go
import copy
from plotly.subplots import make_subplots

# 사용자 정의 simglucose 경로 설정
sys.path.insert(0, './simglucose')

from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.controller.base import Action

components.html("""
    <script>
        window.scrollTo({ top: 0, behavior: 'smooth' });
    </script>
""", height=0)

def plot_static(fig, **kwargs):
    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True, **kwargs})
    
if st.session_state.get("trigger_scroll", False):
    components.html("""
        <script>
            window.scrollTo({ top: 0, behavior: 'smooth' });
        </script>
    """, height=0)
    st.session_state.trigger_scroll = False  # 플래그 초기화
    
def show_section_info(df, env, section_index):
    STEP_PER_SECTION = 160  # 3분 간격 × 160 = 480분 = 8시간
    start = section_index * STEP_PER_SECTION
    end = start + STEP_PER_SECTION

    # ⏱ 정확한 시간 샘플링 (df 기반)
    section_start_time = pd.to_datetime(df["Time"].iloc[start])
    section_end_time = pd.to_datetime(df["Time"].iloc[end - 1])

    # 🩸 현재 혈당
    try:
        if hasattr(env, "history") and len(env.history) > 0:
            last_obs = env.history[-1]
            if hasattr(last_obs, "observation"):
                current_bg = last_obs.observation[0]
            elif isinstance(last_obs, dict) and "observation" in last_obs:
                current_bg = last_obs["observation"][0]
            else:
                current_bg = df["BG"].iloc[start]
        else:
            current_bg = df["BG"].iloc[start]
    except Exception:
        current_bg = df["BG"].iloc[start]
    current_bg = round(current_bg, 1)

    # 🍽 식사 정보 + 음식 예시 추가 (다양한 종류로 확장)
    meal_df = df.iloc[start:end][df["CHO"] > 0]
    if not meal_df.empty:
        meal_events = []
        for _, row in meal_df.iterrows():
            time = pd.to_datetime(row["Time"]).strftime("%H:%M")
            cho = round(row["CHO"], 1)
            
            # 탄수화물 → 음식 예시 및 칼로리 대략 추정
            if cho < 10:
                food = "딸기 한 줌 🍓 / 우유 1컵 🥛"
            elif cho < 20:
                food = "바나나 1개 🍌 / 고구마 반 개 🍠"
            elif cho < 30:
                food = "식빵 1.5장 🍞 / 그래놀라 요거트 🥣"
            elif cho < 40:
                food = "공기밥 반 공기 🍚 / 토스트 세트 🍳"
            elif cho < 55:
                food = "라면 1개 🍜 / 김밥 1줄 🍙"
            elif cho < 70:
                food = "한식 도시락 🍱 / 떡볶이 + 순대 🍢"
            else:
                food = "햄버거 세트 🍔🍟 / 피자 2조각 🍕"

            estimated_kcal = int(cho * 4)  # 탄수화물 1g = 약 4 kcal

            meal_events.append(
                f"{time}쯤에 {food}를 먹었어요. \n 탄수화물 약 {cho}g → 약 {estimated_kcal} kcal 정도 됩니다."
            
            )

        meal_info = "🍽 식사 기록 요약:\n- " + "\n- ".join(meal_events)
    else:
        meal_info = "🥛 공복 상태: 해당 구간에 식사 없음"

    # 🏃 활동 정보
    section_map = {
        0: "🌅 수면 시간",
        1: "🏃‍♂️ 일반 활동",
        2: "🌇 저녁 전 안정기"
    }
    activity = section_map.get(section_index, "📌 일반 구간")

    # 💉 권장 인슐린 계산
    target_bg = 110
    gf = 50
    icr = 10
    meal_carb = df.iloc[start:end]["CHO"].sum()
    correction = max((current_bg - target_bg), 0) / gf
    meal_insulin = meal_carb / icr
    recommended_bolus = round(correction + meal_insulin, 2)

    # 볼루스 주입 시점 계산 (식사 30분 전, CHO >= 30g인 경우만)
    bolus_time_info = ""
    main_meals = df.iloc[start:end][df["CHO"] >= 10]
    if not main_meals.empty:
        first_meal_time = pd.to_datetime(main_meals["Time"].iloc[0])
        bolus_time = first_meal_time - pd.Timedelta(minutes=30)
        df_section = df.iloc[start:end].reset_index(drop=True)
        df_section["Time"] = pd.to_datetime(df_section["Time"])
        bolus_idx = (df_section["Time"] - bolus_time).abs().idxmin()
        bolus_time_info = f"🍚 주요 식사 감지됨: {first_meal_time.strftime('%H:%M')}\n💉 볼루스 인슐린은 `{bolus_time.strftime('%H:%M')}`에 1회 주입)"

    st.subheader("1. 먼저 환자의 지금 상황을 파악해 보세요.")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("CGM.png", caption="혈당 측정기", use_container_width=True)
    with col2:
        st.success(f"⏱ 이번 시간은: {section_start_time.strftime('%H:%M')} ~ {section_end_time.strftime('%H:%M')}이며, {activity}입니다.")
        st.success(f"🩸 센서로 **측정된 현재 혈당** `{current_bg} mg/dL`입니다")
        st.success("혈당의 권장 범위는 70~180 mg/dL입니다.")
        st.success(meal_info)

    st.subheader("2. 혈당을 낮추는데 필요한 인슐린 양을 계산해 보세요.")
    with st.expander("📘 인슐린 주입 기준 보기", expanded=False):
        st.markdown("""
        #### 💉 인슐린 주입은 이렇게 해요

        - **볼루스 인슐린**: 식사를 할 때 혈당이 급격히 오르지 않도록 한 번에 주사하는 인슐린이에요.  
        주로 **식사 30분 전**에 한 번 맞아요.

            - 얼마나 맞아야 할지는 먹는 **탄수화물 양**과 현재 혈당에 따라 달라져요.
            - 계산 방법은 다음과 같아요:
                ```
                볼루스 인슐린 양 = (먹은 탄수화물 양 ÷ ICR) + 보정량
                ```
                - 예: 밥이나 빵 등 탄수화물 60g을 먹을 때,  
                ICR(탄수화물 10g당 1단위)이면 → `60 ÷ 10 = 6단위`
                - 보정량: 현재 혈당이 너무 높을 경우, 추가로 조금 더 주입해야 해요.  
                예: (현재 혈당 - 목표 혈당) ÷ 감도 계수(GF)

        - **기저 인슐린**: 식사와 상관없이 하루 종일 천천히 나오는 인슐린이에요.  
        보통 **3분마다 아주 조금씩** 주입되며, 혈당을 일정하게 유지하는 데 도움을 줘요.

            - 예를 들어, 한 번에 `0.02단위`씩, 8시간 동안 계속 주입하면:
                ```
                0.02 × 160번(8시간 기준) = 총 3.2단위
                ```

        ---

        ⚠️ 참고: 인슐린 주입량은 사람마다 다르며,  
        이 시뮬레이터는 실제 치료가 아닌 **학습을 위한 도구**예요.
        """)

    if bolus_time_info:
        st.success(bolus_time_info)
    st.info(f"""
    ### 💉 권장 인슐린 계산 정보
    - 목표 혈당: {target_bg} mg/dL산
    - 감도 계수(GF): {gf}, ICR: {icr}
    - 예상 식사량: **{meal_carb:.1f} g 탄수화물**
    - 권장 볼루스 인슐린: **{recommended_bolus}**
    - 권장 기저 인슐린: **0.03 ~ 0.04**
    """)

def analyze_glucose_events(bg_series, time_series):
    df_g = pd.DataFrame({"time": time_series, "bg": bg_series})
    df_g["status"] = "정상"
    df_g.loc[df_g["bg"] < 70, "status"] = "저혈당"
    df_g.loc[df_g["bg"] > 180, "status"] = "고혈당"

    df_g["group"] = (df_g["status"] != df_g["status"].shift()).cumsum()
    events = df_g[df_g["status"] != "정상"].groupby("group")

    messages = []
    for _, group in events:
        status = group["status"].iloc[0]
        t_start = group["time"].iloc[0].strftime("%H:%M")
        t_end = group["time"].iloc[-1].strftime("%H:%M")
        messages.append(f"- **{t_start} ~ {t_end}** 사이에 **{status}** 발생")

    return messages, df_g

def summarize_today(basal_list, bolus_list, meal_total, bg_series):
    summary = []

    # 총량 계산
    basal_total = sum(basal_list)
    bolus_total = sum(bolus_list)
    insulin_total = basal_total + bolus_total

    # # 1. 인슐린 총량 평가
    # if insulin_total > 10:
    #     summary.append("💉 인슐린을 전반적으로 많이 사용했습니다.")
    # elif insulin_total < 5:
    #     summary.append("💉 인슐린 사용량이 다소 부족했습니다.")
    # else:
    #     summary.append("💉 인슐린 용량은 적절한 수준이었습니다.")

    # 2. 식사량 평가
    if meal_total > 150:
        summary.append("🍚 오늘 섭취한 탄수화물 양이 많아 혈당 조절이 어려웠을 수 있습니다.")
    elif meal_total < 50:
        summary.append("🥛 식사량이 적어 저혈당 위험이 있을 수 있습니다.")
    else:
        summary.append("🥗 적절한 식사량이 유지되었습니다.")

    # 3. 혈당 패턴 평가
    hypo = sum(bg < 70 for bg in bg_series)
    hyper = sum(bg > 180 for bg in bg_series)

    if hypo > 5:
        summary.append("⚠️ 저혈당이 여러 차례 발생했습니다. 기저 인슐린을 줄이는 것이 좋겠습니다.")
    elif hyper > 5:
        summary.append("⚠️ 고혈당이 자주 발생했습니다. 식사량을 조절하거나 볼루스 인슐린을 늘려야 할 수 있습니다.")
    else:
        summary.append("✅ 혈당이 안정적으로 유지되었습니다.")

    # 종합 제안
    if hyper > 5 and meal_total > 150:
        summary.append("📌 식사량을 줄이거나 식후 가벼운 운동을 병행하면 혈당 조절에 도움이 됩니다.")
    elif hypo > 5 and insulin_total > 10:
        summary.append("📌 인슐린 용량을 줄이고 간식을 적절히 배분하는 것이 필요합니다.")
    elif 0 < hypo <= 5 or 0 < hyper <= 5:
        summary.append("📌 혈당 조절이 거의 잘 되었으나 약간의 보완 여지가 있습니다.")

    return "\n".join(summary)

# 세션 상태 초기화
if "step" not in st.session_state:
    st.session_state.step = 0
if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = None

st.title("🩺 인슐린 제어 시뮬레이터")

# STEP 0: 환자 선택
if st.session_state.step == 0:

    st.markdown("""
    ### 👨‍⚕️ 인슐린 제어 시뮬레이터 소개

    가상의 제1형 당뇨병 환자를 대상으로,  
    **사용자가 직접 인슐린 주입량을 설정**하고  
    **AI 제어와 비교**해볼 수 있는 학습형 시뮬레이터입니다.

    ---

    #### 🎯 체험 목적 요약
    - 당뇨병 환자의 혈당 조절 어려움 **간접 체험**
    - 인슐린 **타이밍과 용량**의 중요성 학습
    - **기저 인슐린과 볼루스 인슐린**의 역할 이해
    """)
    st.image("diabetes.png", use_container_width=True)

    st.markdown("## 💡 왜 혈당 조절이 어려울까요?")

    st.markdown("""
    당뇨병 환자에게 인슐린 조절은 매일 반복되는 과제입니다.  
    그중에서도 **식사와 혈당 측정**은 생명과 직결된 요소입니다.
    """)
    
    # ⏱ 혈당 조절 관련 팩트 카드
    with st.expander("🩸 혈당 스파이크란?"):
        st.markdown("""
        #### 📊 혈당 조절, 왜 중요할까요?

        - 🍚 건강한 사람은 식사 후 혈당이 **140mg/dL 이하**로 조절됩니다.  
        - 그러나 당뇨 환자는 쉽게 **180mg/dL 이상**으로 올라가며 이를 **혈당 스파이크**라고 부릅니다.
        - 💥 이 상태가 반복되면 **신장, 신경, 혈관계에 심각한 손상**을 줄 수 있습니다.
        - ⏱ 인슐린 주입이 30분만 늦어져도 혈당 조절은 큰 영향을 받습니다.

        ---
        🤖 AI는 정확한 시점과 용량을 계산해 최적의 인슐린 제어를 시도합니다.  
        🧑 사용자도 이를 직접 조절해보며 **혈당 반응을 체험**할 수 있습니다.
        """)

    # 💉 인슐린 타입 설명
    with st.expander("💉 인슐린 이란?"):
        st.markdown("""
        인슐린은 몸 안에서 혈당을 조절해주는 생명에 꼭 필요한 호르몬입니다.
        당뇨병 환자는 이 인슐린을 제대로 만들거나 활용하지 못해,
        식사 후 혈당이 급격히 올라가고 몸에 큰 부담을 주게 됩니다.

        특히 제1형 당뇨병 환자는 몸속에서 인슐린을 전혀 만들지 못하기 때문에,
        하루에도 여러 번 주사나 인슐린 펌프를 통해 외부에서 직접 주입해야 합니다.

        🩸 이 주사는 단순히 불편한 것을 넘어
        매일 반복되는 고통과 스트레스를 동반합니다.
        "지금 얼마나 넣어야 할까?", "혹시 저혈당이 올까?"라는 불안감은
        환자들의 일상에 늘 그림자처럼 따라붙습니다.

        이 시뮬레이터는
        그들의 하루를 조금이나마 체험해보고,
        AI의 도움으로 어떻게 부담을 줄일 수 있을지 함께 고민해보기 위해 만들어졌습니다.
                    
        | 구분 | 기저 인슐린 (Basal) | 식전 볼루스 인슐린 (Bolus) |
        |------|--------------------|-----------------------------|
        | 역할 | 공복 혈당 조절     | 식후 혈당 급등 억제         |
        | 타이밍 | 하루 1~2회 지속 주입 | 식사 30분 전               |
        | 작용 시간 | 느리고 지속적     | 빠르고 단기적              |

        """, unsafe_allow_html=True)

    with st.expander("🍽 식사는 왜 신중해야 하나요?"):
        st.markdown("""
        - **먹는 음식이 곧 혈당**입니다.  
        - 같은 음식도 **시간, 양, 활동량**에 따라 혈당 반응이 달라집니다.
        - 식사 전 인슐린(볼루스)을 **적절한 양으로, 미리** 주입하지 않으면  
        → 혈당이 **180mg/dL 이상으로 급등(스파이크)**할 수 있습니다.
        """)

    with st.expander("🩸 혈당 측정 어떻게 하나요?"):
        st.markdown("""
        - 당뇨병 환자는 하루에도 **여러 번 혈당을 측정**합니다.
        - 이는 단순한 숫자가 아니라,  
        **“지금 내 몸은 안전한가?”를 확인하는 생존의 도구**입니다.
        - 측정 없이 인슐린을 맞으면 → **저혈당 쇼크**나 **과다투여 위험** 발생
        """)

    # 📈 TIR 설명
    with st.expander("📈 TIR(Time in Range)이란?"):
        st.markdown("""
        - 혈당이 **70~180 mg/dL** 범위 내에 머무는 시간 비율입니다.  
        - TIR이 높을수록 혈당이 안정적으로 유지되고 있다고 볼 수 있습니다.  
        - 본 시뮬레이터에서는 **AI vs 사용자** TIR을 비교하여 인슐린 전략의 효과를 확인할 수 있습니다.
        """)

    st.markdown("---")
    st.success("✅ 아래에서 시뮬레이션할 환자를 선택한 후, 다음 단계로 이동하세요.")

    st.subheader("환자 선택")
    patient_name = st.selectbox("시뮬레이션할 환자를 선택하세요:", [
        "adult#001", "adult#002", "adult#003","adult#004", "adult#005",
        "adolescent#001", "adolescent#002", "adolescent#003","adolescent#004", "adolescent#005",
    ])
    
    if st.button("다음 단계로"):
        st.session_state.selected_patient = patient_name
        st.session_state.csv_file = f"{patient_name}_100_500.csv"
        st.session_state.step = 1
        st.rerun()

# STEP 1: 데이터 불러오기
elif st.session_state.step == 1:
    df = pd.read_csv(f"data/{st.session_state.csv_file}")
    df["Time"] = pd.to_datetime(df["Time"])
    st.session_df = df
    st.write(f"✅ {st.session_state.selected_patient} 데이터 로딩 완료")

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

    st.subheader("혈당 조절 과정은 다음과 같습니다.")
    st.markdown("""
                - 본 과정에서는 혈당 측정과 식사량 계산 정보를 제공합니다.
                - 체험 중에는 인슐린 주입만 고려하고 혈당 변화만 확인 하면 됩니다.
                - 적절한 인슐린량은 시시각각 변화합니다. 상황에 맞게 잘 판단하세요.
                """)
    
    st.image("how.png", use_container_width=True)

    with st.expander("ℹ️ 혈당 조절 가이드", expanded=False):
        st.markdown("""
            ### 🔄 혈당 조절은 이렇게 진행돼요!

            1. **현재 혈당 확인하기**  
            - 혈당 측정기를 통해 지금 내 혈당이 어느 정도인지 확인해요.

            2. **식사 시간과 양 체크하기**  
            - 최근 식사를 했는지, 얼마나 먹었는지 (특히 밥·빵·면 같은 탄수화물!) 확인해요.

            3. **인슐린 양 계산하기 (추천값 제공)**  
            - 목표 혈당: `110 mg/dL`  
            - 혈당이 얼마나 쉽게 떨어지는지 나타내는 수치: `감도 계수(GF) = 50`  
            - 탄수화물 10g당 필요한 인슐린: `1.0 단위`  
            ➡️ 위 정보를 바탕으로 **추천 인슐린 주사량**을 알려드려요!

            4. **인슐린 주사하기 (2가지 방식)**  
            - **기저 인슐린**: 하루 중 일정하게 천천히 나오는 인슐린 (예: 8시간 동안 조금씩 주입)  
            - **볼루스 인슐린**: 식사 30분 전에 한 번에 주사해서 혈당을 조절해요

            5. **시뮬레이션 실행!**  
            - 설정한 인슐린 주입량에 따라 8시간 동안의 혈당 변화를 확인해볼 수 있어요.

            6. **결과 확인하고 피드백 받기**  
            - 혈당이 정상 범위(70~180 mg/dL)에 들어갔는지 확인해요  
            - 고혈당이나 저혈당이 있었던 시간도 알려드려요  
            - 인슐린 용량을 더 늘릴지 줄일지, 다음 조절 방향에 대한 팁도 드려요!
            """)


    if st.button("➡️ 시뮬레이션 시작"):
        st.session_state.step = 21
        st.rerun()

# STEP 21~23: 각 구간별 시뮬레이션
for seg in [1, 2, 3]:
    if st.session_state.step == 20 + seg:
        df = st.session_df
        ai_df = df.iloc[:480].reset_index(drop=True)
        init_bg = ai_df["BG"].iloc[0]

        if seg == 1 and "env_user" not in st.session_state:
            sensor = CGMSensor.withName("Dexcom")
            pump = InsulinPump.withName("Insulet")
            patient = T1DPatient.withName(st.session_state.selected_patient, init_state=init_bg)
            scenario = RandomScenario(start_time=datetime.datetime.now(), seed=42)
            env_user = T1DSimEnv(patient=patient, sensor=sensor, pump=pump, scenario=scenario)
            env_user.reset()
            st.session_state.env_user = env_user

        env = st.session_state.env_user
        # st.subheader(f"🧪 구간 {seg}: {(seg - 1) * 6}~{seg * 6} 시간")

        dose_key = f"dose{seg}"
        bg_key = f"bg_user{seg}"
        env_init_key = f"env_state_{seg}"
        env_result_key = f"env_result_{seg}"

        # 예: STEP 21, 22, 23에서 각 구간 정보 보여줄 때
        section_index = st.session_state.step - 21
        show_section_info(df, env, section_index)


        st.markdown("슬라이더를 움직여 인슐린 주입량을 조절해 보세요")
        # dose = st.slider(f"볼루스 인슐린 (식사 30분전 주입)", 0.0, 5.0, 1.0, 0.1, key=dose_key)
        dose = st.slider("볼루스 인슐린 (식사전 주요입)", 0.0, 5.0, st.session_state.get("dose1", 1.0), key="dose1")
        basal = st.slider("기저 인슐린 (8시간 동안 주입)", 0.0, 0.05, st.session_state.get("dose_basal", 0.02), 0.001, key=f"basal{seg}")
        
        # 💉 총 인슐린 투여량 계산
        total_basal = round(basal * 160, 2)  # 160 스텝 동안의 총 기저 인슐린
        total_bolus = dose                   # 한 번에 주입
        total_insulin = round(total_basal + total_bolus, 2)

        # # 💬 사용자에게 보여주기
        # st.markdown(f"""
        # 🔢 **8시간 총 인슐린 투여량**  
        # - 💧 기저 인슐린: `{basal} × 160 = {total_basal} 단위`  
        # - 💉 볼루스 인슐린: `{total_bolus} 단위`  
        # - ✅ **총 투여량**: `{total_insulin} 단위`
        # """)
        st.subheader("3. 인슐린 주입 후 환자의 혈당 변화를 확인해 보세요.")
        if env_init_key not in st.session_state:
            st.session_state[env_init_key] = copy.deepcopy(env)

        if st.button(f"시뮬레이션 {seg} 실행"):
            env = copy.deepcopy(st.session_state[env_init_key])
            result = []

            # 1️⃣ 식사 시점 탐지 및 볼루스 주입 시점 설정
            section_df = df.iloc[seg * 160 : (seg + 1) * 160]
            meal_times = section_df[section_df["CHO"] >= 10].index.tolist()
            bolus_step = None
            if meal_times:
                meal_step = meal_times[0] - section_df.index[0]
                bolus_step = max(meal_step - 10, 0)

            for t in range(160):
                bolus = dose if bolus_step is not None and bolus_step == t else 0.0
                obs, _, _, _ = env.step(Action(basal=basal, bolus=bolus))
                result.append(obs[0])
            st.session_state[bg_key] = result
            st.session_state[env_result_key] = copy.deepcopy(env)

            # ⏱ x축: 시작 시간 + 3분 간격 × 스텝
            start_time = datetime.datetime.strptime("00:00", "%H:%M") + datetime.timedelta(minutes=seg * 160 * 3)
            time_range = [start_time + datetime.timedelta(minutes=3 * i) for i in range(160)]

            # 🥗 현재 구간에 해당하는 식사량 시계열
            section_df = df.iloc[seg * 160 : (seg + 1) * 160].reset_index(drop=True)
            meal_series = section_df["CHO"].tolist()

            # 📈 복합 시각화
            fig = go.Figure()

            # ✅ 1. 정상 범위 음영 (70~180 mg/dL)
            fig.add_shape(
                type="rect",
                xref="x", yref="y",
                x0=time_range[0], x1=time_range[-1],
                y0=70, y1=180,
                fillcolor="green",
                opacity=0.2,
                layer="below",
                line_width=0,
            )


            # 1️⃣ 혈당 선 그래프
            fig.add_trace(go.Scatter(
                x=time_range,
                y=result,
                mode="lines",
                name="혈당 (mg/dL)",
                line=dict(color="red"),
                yaxis="y1"
            ))

            # 2️⃣ 식사량 막대 그래프
            fig.add_trace(go.Bar(
                x=time_range,
                y=meal_series,
                name="식사량 (CHO g)",
                marker_color="lightblue",
                opacity=0.6,
                yaxis="y2"
            ))

            # 📐 그래프 레이아웃
            fig.update_layout(
                title=f"구간 {seg} 혈당 및 식사량",
                xaxis=dict(
                    title="시간 (시:분)",
                    tickformat="%H:%M",
                    tickangle=45
                ),
                yaxis=dict(
                    title="혈당 (mg/dL)",
                    range=[40, max(result) + 20],
                    side="left"
                ),
                yaxis2=dict(
                    title="식사량 (g)",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                    range=[0, max(meal_series) + 10 if any(meal_series) else 10]
                ),
                legend=dict(x=0.01, y=1.1, orientation="h"),
                bargap=0.1
            )

            # 📊 렌더링
            # st.plotly_chart(fig, use_container_width=True)
            plot_static(fig)

            # 📊 혈당 결과 해석
            st.markdown("#### 🔍 혈당 결과 해석")
            bg_final = result[-1]
            if bg_final < 70:
                st.warning(f"⚠️ 최종 혈당이 {bg_final:.1f} mg/dL로 저혈당입니다. 인슐린 용량을 줄여보세요.")
            elif bg_final > 180:
                st.warning(f"⚠️ 최종 혈당이 {bg_final:.1f} mg/dL로 고혈당입니다. 인슐린 용량을 늘려볼 수 있습니다.")
            else:
                st.success(f"✅ 최종 혈당 {bg_final:.1f} mg/dL — 안정적인 범위입니다.")

            events, _ = analyze_glucose_events(result, time_range)
            st.subheader("🩸 혈당 이상 구간 요약")
            if events:
                for msg in events:
                    st.markdown(msg)
            else:
                st.success("✅ 모든 시간대에서 혈당이 정상 범위(70~180 mg/dL)를 유지했습니다.")

            
        # if st.button(f"🔁 구간 {seg} 다시 설정"):
        #     if bg_key in st.session_state:
        #         del st.session_state[bg_key]

        if st.button("➡️ 다음 구간으로"):
            st.session_state.env_user = copy.deepcopy(st.session_state[env_result_key])
            st.session_state.dose_basal = basal
            st.session_state.step += 1
            st.rerun()


# STEP 24: 결과 통합 시각화
if st.session_state.step == 24:
    st.subheader("✅ 전체 시뮬레이션 결과 요약")

    
    fig = go.Figure()
    full_bg = []
    full_bolus = []
    full_basal = []

    # 샘플 타임스탬프 생성 (3분 간격, 총 480개: 24시간 분량)
    start_time = datetime.datetime.strptime("00:00", "%H:%M")
    time_range = [start_time + datetime.timedelta(minutes=3 * i) for i in range(480)]

    # 데이터 로드
    df = pd.read_csv(f"data/{st.session_state.csv_file}")
    df["Time"] = pd.to_datetime(df["Time"])

    for i in range(1, 4):
        bg_key = f"bg_user{i}"
        dose_key = f"dose{i}"
        if bg_key in st.session_state:
            full_bg.extend(st.session_state[bg_key])
            dose_value = st.session_state.get(dose_key, 0.0)
            full_bolus.extend([dose_value] * 160)
            full_basal.extend([st.session_state.get("dose_basal", 0.02)] * 160)

    meal_total = df.iloc[:480]["CHO"].sum()

    # 요약 생성
    st.markdown("### 📊 오늘의 혈당 제어 요약")
    st.success(summarize_today(full_basal, full_bolus, meal_total, full_bg))

    # AI 제어 결과 불러오기
    ai_df = st.session_df.iloc[:480].reset_index(drop=True)
    ai_bg = ai_df["BG"].tolist()

    # 혈당 비교 시각화
    fig = go.Figure()
        
    fig_combined = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. 정상 혈당 범위 음영
    fig_combined.add_shape(
        type="rect",
        xref="x", yref="y",
        x0=time_range[0], x1=time_range[-1],
        y0=70, y1=180,
        fillcolor="green", opacity=0.2,
        layer="below", line_width=0
    )

    # 2. 사용자 혈당
    fig_combined.add_trace(go.Scatter(
        x=time_range, y=full_bg,
        mode="lines", name="사용자 혈당",
        line=dict(color="blue")
    ), secondary_y=False)

    # 3. AI 혈당
    fig_combined.add_trace(go.Scatter(
        x=time_range, y=ai_bg,
        mode="lines", name="AI 혈당",
        line=dict(color="gray", dash="dot")
    ), secondary_y=False)

    # # 4. Bolus 인슐린
    # fig_combined.add_trace(go.Scatter(
    #     x=time_range, y=full_bolus,
    #     mode="lines", name="Bolus 인슐린",
    #     line=dict(color="red", width=1, dash="dash")
    # ), secondary_y=True)

    # # 5. Basal 인슐린
    # fig_combined.add_trace(go.Scatter(
    #     x=time_range, y=full_basal,
    #     mode="lines", name="Basal 인슐린",
    #     line=dict(color="orange", width=1)
    # ), secondary_y=True)

    # 6. 레이아웃 설정
    fig_combined.update_layout(
        # title="AI vs 사용자 혈당",
        xaxis_title="시간",
        yaxis_title="혈당 (mg/dL)",
        legend=dict(x=0, y=1.15, orientation="h"),
        height=600
    )

    # 보조 y축 설정
    fig_combined.update_yaxes(title_text="혈당 (mg/dL)", secondary_y=False)
    # fig_combined.update_yaxes(title_text="인슐린 (U)", secondary_y=True)

    # 시각화 출력
    plot_static(fig_combined)
   
    # 3. TIR 계산 및 막대 시각화
    st.markdown("하루동안 혈당이 얼마나 정상범위에 머물렀는지 보여줍니다.")
    def compute_tir(bg_series):
        in_range = np.logical_and(np.array(bg_series) >= 70, np.array(bg_series) <= 180)
        return 100 * np.sum(in_range) / len(bg_series)

    tir_ai = compute_tir(ai_bg)
    tir_user = compute_tir(full_bg)

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

    # st.plotly_chart(fig_tir, use_container_width=True)
    plot_static(fig_tir)

    st.subheader("📊 TIR (Time in Range: 70~180 mg/dL)")
    st.write(f"✅ **AI TIR**: {tir_ai:.2f}%")
    st.write(f"🧑‍⚕️ **사용자 TIR**: {tir_user:.2f}%")

    def compute_variability(bg_series):
        bg_array = np.array(bg_series)
        avg = np.mean(bg_array)
        std = np.std(bg_array)
        cv = (std / avg) * 100
        return avg, std, cv
    
    # 계산
    avg_ai, std_ai, cv_ai = compute_variability(ai_bg)
    avg_user, std_user, cv_user = compute_variability(full_bg)

        # 기존 결과 비교 메시지 대체
    st.subheader("🏁 결과 요약")

    if tir_user > tir_ai and cv_user < cv_ai:
        st.success("🎯 사용자 제어가 AI보다 TIR도 높고 혈당 변동성도 낮아 우수한 제어를 보여주었습니다.")
    elif tir_user > tir_ai and cv_user > cv_ai:
        st.info(f"📈 사용자의 TIR은 높지만 변동성이 큽니다. (CV {cv_user:.1f}% > {cv_ai:.1f}%)")
    elif tir_user < tir_ai and cv_user < cv_ai:
        st.warning(f"🤖 AI의 TIR은 높지만, 사용자의 혈당 변동성이 더 낮아 안정적인 제어를 보였습니다.")
    elif tir_user < tir_ai and cv_user > cv_ai:
        st.error("⚠️ AI 제어가 TIR과 혈당 안정성 모두에서 더 우수했습니다.")
    else:
        st.info("⚖️ 사용자와 AI가 유사한 수준의 혈당 제어 성능을 보여주었습니다.")

    # 표 형태 요약
    st.subheader("📊 혈당 변동성 비교")

    st.markdown(f"""
    | 구분 | 평균 혈당 | 표준편차 (SD) | 변동계수 (CV%) |
    |------|------------|----------------|----------------|
    | **AI** | {avg_ai:.1f} mg/dL | {std_ai:.1f} | {cv_ai:.1f}% |
    | **사용자** | {avg_user:.1f} mg/dL | {std_user:.1f} | {cv_user:.1f}% |
    """, unsafe_allow_html=True)
    
    st.subheader("수고하셨습니다! 다음엔 더 나은 결과를 기대해볼까요?")
    st.subheader("다른 환자도 직접 도전해보세요.")
    if st.button("✅ 시뮬레이션 완료 → 처음으로"):
        for key in list(st.session_state.keys()):
            if key.startswith("bg_user") or key.startswith("env_") or key.startswith("dose"):
                del st.session_state[key]
        st.session_state.step = 0
        st.rerun()