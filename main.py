import streamlit as st
import numpy as np
import pandas as pd
from korean_name_generator import namer
from faker import Faker

# 남성 키 생성 함수
def generate_men_heights(mean=171, std_dev=5, min_height=150, max_height=230, size=100):
    heights = []
    while len(heights) < size:
        height = np.random.normal(mean, std_dev)
        if min_height <= height <= max_height:
            heights.append(round(height, 1))  # 소수점 첫째 자리까지 반올림
    return heights

# 여성 키 생성 함수
def generate_women_heights(mean=158, std_dev=5, min_height=130, max_height=230, size=100):
    heights = []
    while len(heights) < size:
        height = np.random.normal(mean, std_dev)
        if min_height <= height <= max_height:
            heights.append(round(height, 1))  # 소수점 첫째 자리까지 반올림
    return heights

# 남성 몸무게 생성 함수
def generate_men_weights(mean=72, std_dev=10, min_weight=35, max_weight=150, size=100):
    weights = []
    while len(weights) < size:
        weight = np.random.normal(mean, std_dev)
        if min_weight <= weight <= max_weight:
            weights.append(round(weight, 1))  # 소수점 첫째 자리까지 반올림
    return weights

# 여성 몸무게 생성 함수
def generate_women_weights(mean=55, std_dev=5, min_weight=30, max_weight=120, size=100):
    weights = []
    while len(weights) < size:
        weight = np.random.normal(mean, std_dev)
        if min_weight <= weight <= max_weight:
            weights.append(round(weight, 1))  # 소수점 첫째 자리까지 반올림
    return weights

# BMI 계산 함수
def calculate_bmi(weights, heights):
    bmis = []
    for weight, height in zip(weights, heights):
        height_m = height / 100  # cm를 m로 변환
        bmi = weight / (height_m ** 2)
        bmis.append(round(bmi, 1))  # 소수점 첫째 자리까지 반올림
    return bmis

# 성별에 따라 키와 몸무게를 생성하고 BMI를 계산하는 함수
def generate_physical_data(gender, size=100):
    if gender.lower() == '남성':
        heights = generate_men_heights(size=size)
        weights = generate_men_weights(size=size)
    elif gender.lower() == '여성':
        heights = generate_women_heights(size=size)
        weights = generate_women_weights(size=size)
    else:
        raise ValueError("Gender must be '남성' or '여성'")
    
    bmis = calculate_bmi(weights, heights)
    return heights, weights, bmis

# 가상의 한국 이름 생성 함수
def generate_korean_name(gender):
    if gender.lower() == '남성':
        return namer.generate(True)  # 남성 이름 생성
    elif gender.lower() == '여성':
        return namer.generate(False)  # 여성 이름 생성
    else:
        raise ValueError("Gender must be '남성' or '여성'")

# 나이 생성 함수 
def generate_ages(size=100):
    # 20~99세 범위 내에서 가중치를 부여하여 나이 생성
    age_range = np.arange(20, 100)
    weights = np.concatenate([
        np.full(20, 9),  # 20~30대에 가중치 9 부여
        np.full(40, 0.5),  # 31~80대에 가중치 0.5 부여
        np.full(20, 0.5)  # 81~99대에 가중치 0.5 부여
    ])
    ages = np.random.choice(age_range, size=size, p=weights/weights.sum())
    return ages

# 근육량 생성 함수
def generate_muscle_mass(size=100):
    muscle_mass_categories = ['적음', '평균', '많음']
    weights = [0.2, 0.6, 0.2]  # 적음 20%, 평균 60%, 많음 20%
    muscle_mass = np.random.choice(muscle_mass_categories, size=size, p=weights)
    return muscle_mass

# 체형 생성 함수
def classify_body_type(bmis, muscle_masses, genders):
    body_types = []
    for bmi, muscle_mass, gender in zip(bmis, muscle_masses, genders):
        if bmi < 18.5:
            if muscle_mass == '적음':
                body_type = '마른 몸'
            elif muscle_mass == '평균':
                body_type = '날씬한 몸'
            elif muscle_mass == '많음':
                if gender == '여성':
                    body_type = '날씬하고 근육 많은 몸'
                elif gender == '남성':
                    body_type = '잔근육 많은 몸'
        elif 18.5 <= bmi < 24.9:
            if muscle_mass == '적음':
                body_type = '평균적인 몸'
            elif muscle_mass == '평균':
                body_type = '평균적인 몸'
            elif muscle_mass == '많음':
                body_type = '탄탄한 몸'
        elif 25 <= bmi < 29.9:
            if muscle_mass == '적음':
                body_type = '살짝 통통한 몸'
            elif muscle_mass == '평균':
                body_type = '통통한 몸'
            elif muscle_mass == '많음':
                body_type = '근육이 많은 몸'
        else:  # bmi >= 30
            if muscle_mass == '적음':
                body_type = '살짝 뚱뚱한 몸'
            elif muscle_mass == '평균':
                body_type = '뚱뚱한 몸'
            elif muscle_mass == '많음':
                body_type = '근육돼지'
        body_types.append(body_type)
    return body_types

def generate_facial_features(size=100):
    # 얼굴형 생성
    face_shapes = ['둥근형', '타원형', '사각형', '마름모형', '긴형']
    face_shape_weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # 평범한 얼굴형 50%, 나머지 각 10%
    face_shape_list = np.random.choice(face_shapes, size=size, p=face_shape_weights)
    
    # 코 모양 생성
    nose_shapes = ['높은 코', '낮은 코', '매부리 코', '평범한 코']
    nose_shape_weights = [0.15, 0.15, 0.15, 0.55]  # 평범한 코 50%, 나머지 합계 50%
    nose_shape_list = np.random.choice(nose_shapes, size=size, p=nose_shape_weights)
    
    # 입 모양 생성
    lip_shapes = ['얇은 입술', '두꺼운 입술', '평범한 입술']
    lip_shape_weights = [0.25, 0.25, 0.5]  # 평범한 입술 50%, 나머지 합계 50%
    lip_shape_list = np.random.choice(lip_shapes, size=size, p=lip_shape_weights)
    
    # 귀 모양 생성
    ear_shapes = ['작은 귀', '큰 귀', '평범한 귀']
    ear_shape_weights = [0.25, 0.25, 0.5]  # 평범한 귀 50%, 나머지 합계 50%
    ear_shape_list = np.random.choice(ear_shapes, size=size, p=ear_shape_weights)
    
    return face_shape_list, nose_shape_list, lip_shape_list, ear_shape_list

# 눈 특징 생성 함수
def generate_eye_features(size=100):
    eye_colors = ['갈색', '검정']
    eye_color_weights = [0.2, 0.8]  # 갈색 60%, 검정색 40%
    eye_color_list = np.random.choice(eye_colors, size=size, p=eye_color_weights)
    
    eye_sizes = ['작은 눈', '중간 크기 눈', '큰 눈']
    eye_size_weights = [0.1, 0.8, 0.1]  # 중간 크기 눈 50%, 나머지 합계 50%
    eye_size_list = np.random.choice(eye_sizes, size=size, p=eye_size_weights)
    
    eye_shapes = ['동그란 눈', '아몬드형 눈', '갸름한 눈']
    eye_shape_weights = [0.25, 0.5, 0.25]  # 아몬드형 눈 50%, 나머지 합계 50%
    eye_shape_list = np.random.choice(eye_shapes, size=size, p=eye_shape_weights)

    eyelid_types = ['속쌍', '겉쌍', '없음']
    eyelid_weights = [0.25, 0.25, 0.5]  # 속쌍 30%, 겉쌍 40%, 없음 30%
    eyelid_list = np.random.choice(eyelid_types, size=size, p=eyelid_weights)

    eyelid_depths = ['얕은', '짙은']
    eyelid_depth_weights = [0.8, 0.2]  # 얕은 70%, 짙은 30%
    eyelid_depth_list = np.random.choice(eyelid_depths, size=size, p=eyelid_depth_weights)
    
    return eye_color_list, eye_size_list, eye_shape_list, eyelid_list, eyelid_depth_list

# 문신 생성 함수
def generate_tattoos(size=100):
    tattoos = ['없음', '작음', '보통', '많음']
    tattoo_weights = [0.9, 0.05, 0.025, 0.025]  # 없음 80%, 작은 문신 10%, 보통 5%, 많음 5%
    tattoo_list = np.random.choice(tattoos, size=size, p=tattoo_weights)
    
    return tattoo_list

# 손과 발의 특징 생성 함수
def generate_hand_foot_features(size=100):
    hand_sizes = ['작은 손', '중간 크기 손', '큰 손']
    hand_size_weights = [0.15, 0.7, 0.15]  # 중간 크기 손 50%, 나머지 합계 50%
    hand_size_list = np.random.choice(hand_sizes, size=size, p=hand_size_weights)
    
    foot_sizes = ['작은 발', '중간 크기 발', '큰 발']
    foot_size_weights = [0.15, 0.7, 0.15]  # 중간 크기 발 50%, 나머지 합계 50%
    foot_size_list = np.random.choice(foot_sizes, size=size, p=foot_size_weights)
    
    return hand_size_list, foot_size_list

# 왼손잡이/오른손잡이 및 왼발잡이/오른발잡이 생성 함수
def generate_hand_foot_dominance(size=100):
    hand_dominance = np.random.choice(['왼손', '오른손'], size=size, p=[0.1, 0.9])
    foot_dominance = np.array(['왼발' if hand == '왼손' else '오른발' for hand in hand_dominance])
    
    return hand_dominance, foot_dominance



# 대한민국 주요 도시 및 도의 인구 (2024년 인구 기준, 예제 값)
city_population_dict = {
    '서울': 10349312, '부산': 3678555, '인천': 2628000, '대구': 2566540, '대전': 1475221, '광주': 1416938, 
    '울산': 962865, '세종': 230327, '수원': 1242724, '고양': 1073069, '용인': 1031935, '성남': 850731, 
    '부천': 711424, '안산': 650728, '안양': 634367, '남양주': 634596, '화성': 476297, '평택': 365114, 
    '의정부': 479141, '시흥': 158978, '파주': 179923, '군포': 195236, '이천': 196230, '오산': 158978, 
    '하남': 134902, '양주': 106358, '구리': 129319, '안성': 139876, '포천': 106358, '의왕': 126358, 
    '여주': 89281, '양평': 83367, '동두천': 98062, '가평': 75251, '연천': 55415, '춘천': 209746, 
    '원주': 243387, '강릉': 180611, '동해': 101128, '태백': 48962, '속초': 89047, '삼척': 72124, 
    '홍천': 84625, '횡성': 89174, '영월': 58057, '평창': 56634, '정선': 36080, '철원': 23822, 
    '화천': 24027, '양구': 23717, '인제': 23149, '고성': 229896, '양양': 208288, '청주': 139876, 
    '충주': 97749, '제천': 103620, '보은': 72435, '옥천': 74668, '영동': 56634, '증평': 103620, 
    '진천': 72435, '괴산': 74668, '음성': 56634, '천안': 103620, '공주': 72435, '보령': 74668, 
    '아산': 56634, '서산': 100000, '논산': 85000, '계룡': 65000, '당진': 30000, '금산': 150000, 
    '연기': 130000, '부여': 100000, '서천': 90000, '청양': 90000, '홍성': 80000, '예산': 75000, 
    '태안': 70000, '전주': 100000, '군산': 90000, '익산': 85000, '정읍': 80000, '남원': 75000, 
    '김제': 70000, '완주': 100000, '진안': 90000, '무주': 85000, '장수': 80000, '임실': 75000, 
    '순창': 70000, '고창': 100000, '부안': 90000, '목포': 85000, '여수': 80000, '순천': 75000, 
    '나주': 70000, '광양': 100000, '담양': 90000, '곡성': 80000, '구례': 75000, '고흥': 70000, 
    '보성': 100000, '화순': 90000, '장흥': 85000, '강진': 80000, '해남': 75000, '영암': 70000, 
    '무안': 100000, '함평': 90000, '영광': 85000, '장성': 80000, '완도': 75000, '진도': 70000, 
    '신안': 120000, '포항': 110000, '경주': 100000, '김천': 90000, '안동': 50000, '구미': 40000, 
    '영주': 30000, '영천': 20000, '상주': 100000, '문경': 90000, '경산': 85000, '군위': 80000, 
    '의성': 75000, '청송': 70000, '영양': 100000, '영덕': 90000, '청도': 85000, '고령': 80000, 
    '성주': 75000, '칠곡': 70000, '예천': 50000, '봉화': 40000, '울진': 30000, '울릉': 20000, 
    '창원': 120000, '진주': 110000, '통영': 100000, '사천': 90000, '김해': 50000, '밀양': 40000, 
    '거제': 30000, '양산': 20000, '의령': 100000, '함안': 90000, '창녕': 85000, '고성': 80000, 
    '남해': 75000, '하동': 70000, '산청': 120000, '함양': 110000, '거창': 100000, '합천': 90000, 
    '제주': 50000, '서귀포': 40000
}

# 인구 비율 계산
total_population = sum(city_population_dict.values())
city_population_weights = {city: population / total_population for city, population in city_population_dict.items()}

# 주요 도시 리스트
korean_cities = list(city_population_dict.keys())

# 한국 주소 생성 함수
def generate_korean_addresses(size=100):
    addresses = np.random.choice(korean_cities, size=size, p=list(city_population_weights.values()))
    return addresses


# 성격 유형(MBTI) 및 특징 생성 함수
def generate_mbti(size=100):
    mbti_types = ['INTJ', 'INFJ', 'ISTJ', 'ISFJ', 'INTP', 'INFP', 'ISTP', 'ISFP', 
                  'ENTJ', 'ENFJ', 'ESTJ', 'ESFJ', 'ENTP', 'ENFP', 'ESTP', 'ESFP']
    mbti = np.random.choice(mbti_types, size=size)
    return mbti

# MBTI 성격 유형에 따른 특징 생성 함수 (10가지 특징 중 랜덤으로 3개 선택)
def generate_mbti_features(mbti):
    mbti_features = {
        'INTJ': ['분석적', '독립적', '전략적', '결단력 있음', '직설적', '추상적 사고', '목표 지향적', '계획적', '비평적', '비사교적'],
        'INFJ': ['직관적', '헌신적', '이상주의적', '공감적', '통찰력 있음', '조용함', '창의적', '열정적', '복잡함', '비판적'],
        'ISTJ': ['실용적', '신뢰할 수 있음', '체계적', '책임감 있음', '성실함', '보수적', '현실적', '분석적', '조용함', '성실함'],
        'ISFJ': ['협력적', '충실한', '세심함', '친절함', '조용함', '책임감 있음', '헌신적', '현실적', '보수적', '신뢰할 수 있음'],
        'INTP': ['논리적', '창의적', '독창적', '호기심 많음', '개인주의적', '분석적', '비판적', '독립적', '비사교적', '조용함'],
        'INFP': ['이상주의적', '공감적', '개방적', '창의적', '성찰적', '충실한', '비사교적', '직관적', '열정적', '조용함'],
        'ISTP': ['실용적', '즉흥적', '모험적', '논리적', '독립적', '분석적', '차분한', '직설적', '조용함', '냉정함'],
        'ISFP': ['예술적', '적응력 강함', '온화함', '조용함', '비사교적', '충실한', '개인주의적', '창의적', '비평적', '감정적'],
        'ENTJ': ['지도력 강함', '효율적', '자신감', '직설적', '전략적', '목표 지향적', '현실적', '계획적', '분석적', '비판적'],
        'ENFJ': ['외향적', '카리스마 있음', '열정적', '공감적', '조직적', '친절함', '창의적', '헌신적', '통찰력 있음', '현실적'],
        'ESTJ': ['체계적', '실용적', '지도력 강함', '결단력 있음', '직설적', '현실적', '분석적', '신뢰할 수 있음', '계획적', '비평적'],
        'ESFJ': ['친절한', '협력적', '사회적', '외향적', '헌신적', '현실적', '책임감 있음', '조직적', '충실한', '감정적'],
        'ENTP': ['독창적', '재치 있음', '에너지 넘침', '호기심 많음', '논리적', '사교적', '비판적', '현실적', '직설적', '열정적'],
        'ENFP': ['활기찬', '열정적', '창의적', '개방적', '충실한', '공감적', '사교적', '직관적', '이상주의적', '비판적'],
        'ESTP': ['모험적', '현실적', '에너지 넘침', '즉흥적', '직설적', '논리적', '사교적', '분석적', '독립적', '비사교적'],
        'ESFP': ['외향적', '활기찬', '사교적', '즉흥적', '충실한', '현실적', '친절함', '온화함', '창의적', '감정적']
    }
    features = [np.random.choice(mbti_features[typ], 3, replace=False).tolist() for typ in mbti]
    return features

# 학력 생성 함수
def generate_education(size=100, ages=None, genders=None):
    education_levels = ['중졸', '고졸', '대졸', '대학재학중', '대학원생']
    weights = [0.01, 0.35, 0.25, 0.19, 0.03]  # 각 학력 수준에 대한 가중치
    education = []

    for age, gender in zip(ages, genders):
        if age <= 24:
            eligible_levels = education_levels[:2] + education_levels[2:3]  # 중졸, 고졸, 대학재학중, 대학원생
            eligible_weights = [weights[0], weights[1], weights[3]]
        elif gender == '남성' and 19 <= age <= 28:
            eligible_levels = education_levels[:4]  # 중졸, 고졸, 대졸, 대학재학중
            eligible_weights = weights[:4]
        elif gender == '여성' and 19 <= age <= 26:
            eligible_levels = education_levels[:4]  # 중졸, 고졸, 대졸, 대학재학중
            eligible_weights = weights[:4]
        else:
            eligible_levels = education_levels[:3] + education_levels[4:]  # 중졸, 고졸, 대졸, 대학원생
            eligible_weights = weights[:3] + [weights[4]]

        eligible_weights = np.array(eligible_weights) / sum(eligible_weights)  # 확률의 합이 1이 되도록 정규화
        education.append(np.random.choice(eligible_levels, p=eligible_weights))

    return education

# 장점과 단점을 모순되지 않게 선택하는 함수
def generate_advantages_disadvantages(size=100):
    ethical_advantages = [
    '친절함', '정직함', '배려심', '공정함', '사려 깊음', '겸손함', '성실함', '희생정신', '동정심', '기꺼이 돕는',
    '책임감', '충실함', '인내심', '용서하는 마음', '공감 능력', '기꺼이 나누는', '관대함', '신뢰할 수 있음', '예의 바름',
    '존중심', '봉사 정신', '긍정적 태도', '헌신적', '친근함', '자비심', '낙관적', '감사하는 마음', '화합적', '따뜻한 마음',
    '무조건적인 사랑', '희망적', '온화함', '용기 있음', '도덕적', '진정성', '평화로움', '자기 희생', '포용력', '신중함',
    '양심적', '의리', '선의', '착한 심성', '적극적인 도움', '정의로움', '상냥함', '온정', '따뜻한 미소', '이해심', '지원적'
]

    ethical_disadvantages = [
    '무례함', '거짓말쟁이', '이기적', '불공정함', '무관심', '오만함', '불성실함', '이기적 행동', '냉담함', '무신경',
    '무책임함', '불충실함', '참을성 없음', '복수심', '공감 부족', '탐욕스러움', '편협함', '불신', '무례한 태도', '무시',
    '비관적', '지나친 경쟁심', '자만심', '적대적', '거칠음', '배타적', '무감동', '비열함', '속임수', '부도덕함',
    '무절제', '나태함', '의심 많음', '자기 만족', '불순', '게으름', '타인을 비난하는', '자기 본위적', '침울함', '감사할 줄 모르는',
    '고집 센', '반항적', '충동적', '불평', '독선적', '질투심', '불경', '무감각', '욕심 많은', '비협조적'
]
  
    exclusive_pairs = {
    '친절함': '무례함',
    '정직함': '거짓말쟁이',
    '배려심': '이기적',
    '공정함': '불공정함',
    '사려 깊음': '무관심',
    '겸손함': '오만함',
    '성실함': '불성실함',
    '희생정신': '이기적 행동',
    '동정심': '냉담함',
    '기꺼이 돕는': '무신경',
    '책임감': '무책임함',
    '충실함': '불충실함',
    '인내심': '참을성 없음',
    '용서하는 마음': '복수심',
    '공감 능력': '공감 부족',
    '기꺼이 나누는': '탐욕스러움',
    '관대함': '편협함',
    '신뢰할 수 있음': '불신',
    '예의 바름': '무례한 태도',
    '존중심': '무시',
    '봉사 정신': '이기적',
    '긍정적 태도': '비관적',
    '헌신적': '무관심',
    '친근함': '적대적',
    '자비심': '잔인함',
    '낙관적': '비관적',
    '감사하는 마음': '불만족',
    '화합적': '분열적',
    '따뜻한 마음': '냉정함',
    '무조건적인 사랑': '조건적인 사랑',
    '희망적': '절망적',
    '온화함': '거칠음',
    '용기 있음': '겁쟁이',
    '도덕적': '부도덕함',
    '진정성': '거짓',
    '평화로움': '폭력적',
    '자기 희생': '자기 이익',
    '포용력': '배타적',
    '신중함': '충동적',
    '양심적': '양심 없음',
    '의리': '배신',
    '선의': '악의',
    '착한 심성': '악한 심성',
    '적극적인 도움': '방해',
    '정의로움': '불공정함',
    '상냥함': '거칠음',
    '온정': '냉담함',
    '따뜻한 미소': '냉담한 표정',
    '이해심': '이해 부족',
    '지원적': '비협조적'
}
   
    selected_advantages = []
    selected_disadvantages = []

    for _ in range(size):
        # 장점 선택
        advantage = np.random.choice(ethical_advantages, 3, replace=False).tolist()
        selected_advantages.append(advantage)

        # 모순되지 않는 단점 리스트 생성
        invalid_disadvantages = [exclusive_pairs[a] for a in advantage if a in exclusive_pairs]
        valid_disadvantages = [d for d in ethical_disadvantages if d not in invalid_disadvantages]
        disadvantage = np.random.choice(valid_disadvantages, 3, replace=False).tolist()
        selected_disadvantages.append(disadvantage)

    return selected_advantages, selected_disadvantages

# 데이터 프레임 생성 함수
def generate_dataframe(size=100):
    data = []

    for _ in range(size // 2):
        for gender in ['남성', '여성']:
            heights, weights, bmis = generate_physical_data(gender, size=1)
            name = generate_korean_name(gender)
            age = generate_ages(size=1)[0]
            muscle_mass = generate_muscle_mass(size=1)[0]
            body_type = classify_body_type(bmis, [muscle_mass], [gender])[0]
            face_shape, nose_shape, lip_shape, ear_shape = generate_facial_features(size=1)
            eye_color, eye_size, eye_shape, eyelid, eyelid_depth = generate_eye_features(size=1)
            tattoo = generate_tattoos(size=1)
            hand_size, foot_size = generate_hand_foot_features(size=1)
            hand_dominance, foot_dominance = generate_hand_foot_dominance(size=1)
            address = generate_korean_addresses(size=1)[0]
            mbti = generate_mbti(size=1)[0]
            mbti_features = generate_mbti_features([mbti])[0]
            education = generate_education(size=1, ages=[age], genders=[gender])[0]

            # 장점과 단점 생성
            advantages, disadvantages = generate_advantages_disadvantages(size=1)
            selected_advantages = advantages[0]
            selected_disadvantages = disadvantages[0]

            data.append([
                name, gender.capitalize(), age, heights[0], weights[0], bmis[0], muscle_mass, body_type, 
                face_shape[0], nose_shape[0], lip_shape[0], ear_shape[0],
                eye_color[0], eye_size[0], eye_shape[0], eyelid[0], eyelid_depth[0], tattoo[0], 
                hand_size[0], foot_size[0], hand_dominance[0], foot_dominance[0], address,
                mbti, mbti_features, education, selected_advantages, selected_disadvantages
            ])

    df = pd.DataFrame(data, columns=[
        '이름', '성별', '나이', '키', '몸무게', 'BMI', '근육량', '체형', 
        '얼굴형', '코 모양', '입 모양', '귀 모양',
        '눈 색깔', '눈 크기', '눈 모양', '쌍꺼풀', '쌍꺼풀 깊이', '문신', 
        '손 크기', '발 크기', '주로 쓰는 손', '주로 쓰는 발', '출신지',
        'MBTI', '성격특징', '학력', '장점', '단점'
    ])
    return df



# 스트림릿 앱 시작
st.title('가상 한국인 프로필 생성기')

# 사용자 입력
num_pairs = st.number_input('생성할 남녀(쌍) 개수를 입력하세요', min_value=1, max_value=100, value=5)

if st.button('프로필 생성'):
    # 데이터프레임 생성
    df = generate_dataframe(size=num_pairs*2)
    
    # 결과 표시
    st.write(f'{num_pairs}쌍의 가상 한국인 프로필이 생성되었습니다.')
    st.dataframe(df)
    
    # CSV 다운로드 버튼
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="CSV 파일로 다운로드",
        data=csv,
        file_name="가상인물목록.csv",
        mime="text/csv",
    )

st.write('참고: 이 데이터는 완전히 가상의 것이며, 실제 인물과는 관련이 없습니다.')

