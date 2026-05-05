def build_prompt(original_query: str, context_json_string: str) -> str:
    """
    최종 질문과 JSON 컨텍스트 문자열을 받아 API 프롬프트를 생성합니다.
    """
    prompt_template = """You will be given a JSON object as a string which contains a series of related search queries and their retrieved documents ('hits'). Do not make answer from external knowledge. You must make answer inside of Context.
Your main task is to answer the specific 'Question' provided below. Use the entire JSON data as context to formulate your answer, paying close attention to the 'text' fields within the 'hits' lists.

The JSON data has a list of queries. The 'original' query is the one you need to answer. The other queries are supplementary and provide additional context.

If the 'Question' is in Korean, format your answer in Korean as follows:
##제목##

##서론##

##본론##

##결론##

If the 'Question' is in English, format your answer in English as follows, If English then Just write title inside of ##{{Title}}##:
##{{Title}}##

##Introduction##

##Main Body##

##Conclusion##

--- Context ---
{context}

--- Question ---
{query}


<--- Example Start--->

--- Question ---
What are the key features of a sustainability-oriented corporate culture that shape the development and management of sustainable artificial intelligence?	

--- Answer ---
##Sustainability-Oriented Corporate Culture for Sustainable AI## 

##Introduction## 
As organizations increasingly deploy artificial intelligence to address societal and environmental challenges, the underlying corporate culture plays a decisive role in ensuring that AI applications contribute to sustainable development. A sustainability-oriented culture embeds norms, values, and structures that guide the design, deployment, and governance of AI toward long-term ecological health, social equity, and economic resilience. 

##Main Body## 
1. Shared Sustainability Values: Organizations with a sustainability-oriented culture articulate clear environmental and social objectives, aligning AI teams around common purpose beyond short-term profit. These values become part of mission statements, performance metrics, and internal communications, steering AI initiatives toward reducing waste, emissions, or inequalities. 
2. Leadership Commitment and Incentives: Senior executives visibly champion sustainable AI by tying leadership performance reviews and incentive schemes to sustainability indicators—such as carbon reduction goals or fairness metrics—ensuring that resource allocation and strategic priorities reflect sustainable development aims. 
3. Cross-Functional Collaboration: Such cultures foster strong interfaces among data scientists, engineers, ethical specialists, and external stakeholders (e.g., community representatives, NGOs), enabling AI development processes that integrate technical feasibility with environmental impact assessments and social risk evaluations. 
4. Transparency and Accountability: Transparency mechanisms—public reporting on AI’s resource consumption, decision-making criteria, and environmental footprints—combined with internal accountability structures (audit committees, ethics boards) help organizations monitor and correct unsustainable AI practices. 
5. Continuous Learning and Adaptation: A learning-oriented culture promotes regular training on sustainable AI methods, post-deployment reviews of environmental and social outcomes, and iterative improvements based on feedback loops, thereby institutionalizing best practices like energy-efficient algorithms or bias mitigation techniques. 
6. Inclusive Stakeholder Engagement: By proactively involving suppliers, customers, regulators, and affected communities in AI project governance, firms ensure that diverse perspectives shape sustainable AI solutions, anticipate unintended consequences, and build broader support for long-term initiatives. 

##Conclusion## 
A corporate culture that embeds sustainability at every level—from leadership incentives and shared values to transparent practices and continuous learning—provides the essential environment in which sustainable artificial intelligence can flourish. Consequently, companies with such cultural features are better equipped to harness AI’s potential for reducing environmental impact, promoting social equity, and creating resilient economic value over the long term.

--- Question ---
How can the main applications of machine learning methods for early diagnosis of maternal-fetal conditions and their ethical considerations be summarized?	
--- Answer ---
##Machine Learning Applications for Early Maternal-Fetal Diagnosis and Ethical Aspects## 

##Introduction## 
Advances in machine learning have enabled earlier and more accurate detection of maternal-fetal complications—such as preterm birth and abnormal fetal growth—by mining complex biomedical datasets. 
These predictive tools promise to guide timely clinical interventions, but their deployment raises important ethical questions around patient privacy, data governance, and fairness. 

##Main Body## 
Contemporary methods employ supervised learning algorithms (e.g., support vector machines, random forests, neural networks) to analyze multi-modal 
inputs: ultrasound images, biochemical markers, electronic health records, and wearable sensor signals. 
In preterm birth prediction, models integrate demographic factors, cervical measurements, contraction patterns, and inflammatory profiles to achieve high sensitivity in identifying at-risk pregnancies. 
For fetal growth assessment, algorithms process placental perfusion metrics and biometric parameters to flag intrauterine growth restriction or macrosomia. 
Rigorous validation—using cross-validation, external cohorts, and ROC analysis—has demonstrated robust performance, though algorithm generalizability can be limited by cohort heterogeneity. 
Ethical considerations include safeguarding maternal and fetal data privacy, obtaining informed consent for data collection and model use, ensuring transparency and interpretability of model outputs for clinicians and patients, and actively addressing bias stemming from underrepresented subpopulations in training datasets. 

##Conclusion## 
Machine learning holds significant promise for early diagnosis of maternal-fetal conditions by leveraging diverse data sources and advanced predictive models.
To realize clinical benefits responsibly, it is crucial to implement ethical safeguards—ensuring privacy protection, transparent decision-making, and equitable performance across all patient groups.

--- Question ---
초등학교 수준에서 설명 가능한 인공지능 교육의 핵심 요소와 활용 방안을 어떻게 요약할 수 있나요?
--- Answer ---
##제목## 

초등학교 수준에서 설명 가능한 인공지능 교육의 핵심 요소 및 구성주의적 활용 방안 

##서론## 
인공지능(AI)이 빠르게 일상에 스며들며 AI 원리를 단순히 소비하는 단계를 넘어 스스로 이해하고 설명할 수 있는 능력이 중요해졌다. 특히 초등학생은 문제 해결과 비판적 사고의 기초를 형성하는 시기이므로, AI 개념을 쉬운 언어와 실생활 예제로 설명하고 학습자가 직접 개념을 구성해 보는 경험을 제공하는 것이 핵심이다. 

##본론## 
1. 설명 가능한 AI 교육의 핵심 요소 - 인간 중심의 문제 해결 학습: AI가 실제 삶의 문제를 어떻게 도울 수 있는지 사례를 통해 깨닫게 함 - 알고리즘 원리의 직관적 이해: 블록 코드나 그림, 역할 놀이 등을 활용해 입력·처리·출력 과정을 체험 - 해석력 강화: 학습자가 AI 모델의 결과를 인간의 관점에서 설명하도록 유도해 ‘왜’라는 질문을 강조 - 다양한 교수·학습 도구 활용: 시각화 도구, 마이크로비트·스마트 로봇 등 눈에 보이는 매체로 알고리즘 동작 원리를 학습 
2. 구성주의 관점의 보완적 활용 - 학습자 주도 탐구: 학급 내 소그룹 프로젝트로 AI 활용 과제를 스스로 설계·실행하며 개념을 구성 - 경험 중심 통합: 진화생물학·뇌과학·시스템 이론 등 다양한 배경 지식을 놀이·토론 활동에 녹여 융합적 이해 촉진 - 협력적 피드백: 동료 간 설명·시연 과정을 통해 서로 다른 관점을 공유하며 개념을 수정·발전 

##결론## 
초등학생 대상 설명 가능한 AI 교육은 실생활 사례와 직관적 알고리즘 체험을 통해 해석력과 문제 해결 능력을 동시에 키우며, 구성주의적 수업 설계로 학습자 스스로 지식을 구성하는 과정을 지원할 때 더욱 효과적이다. 이러한 접근은 학생들이 AI 원리를 바탕으로 비판적·창의적 사고를 확장하도록 돕는다.

--- Question ---
생성형 AI 콘텐츠의 가치 유형별 소비자 인식과 AI 사용 공개에 대한 인식 특성을 어떻게 요약할 수 있나요?	

--- Answer ---
##제목## 
생성형 AI 콘텐츠 가치·형식별 소비자 인식과 AI 사용 공개 태도 분석 
##서론## 
생성형 인공지능(AIGC)은 다양한 미디어 형식으로 자율 생성 능력을 가지지만, 진정성 문제로 규제와 투명성 요구가 커지고 있다. 이에 소비자가 AIGC를 어떻게 인지하고 AI 사용 공개를 어떤 원칙으로 받아들이는지 이해하는 것이 중요하다. 본 연구는 콘텐츠 가치 유형(실용적·쾌락적)과 미디어 형식(글·사진·영상)을 결합한 여섯 가지 시나리오에서 소비자 인식을 분석했다. 
##본론## 
카노 모형을 활용해 각 시나리오의 만족도와 불만족 요인을 측정하고, 인지된 진정성과 기만 여부에 따라 그룹을 구분했다. 실용적 콘텐츠(특히 뉴스)에서는 AI 사용에 대해 부정적 인식이 우세했고, 영상 뉴스에서 그 강도가 가장 컸다. 반면 쾌락적 콘텐츠(영화·드라마)에서는 AI 사용에 대한 관심이 낮거나 오히려 긍정적인 태도가 관찰되었다. AI 사용 공개는 콘텐츠 가치에 관계없이 ‘지켜야 할 원칙’으로 여겨졌으나, 실용적 콘텐츠에서 그 중요성이 더 크게 드러났다. 
##결론## 
소비자는 콘텐츠의 가치 유형과 형식에 따라 AIGC와 AI 사용 공개에 상이한 태도를 보인다. 뉴스 같은 정보성 콘텐츠에서는 투명성과 진정성 확보 노력이 필수적하며, 쾌락적 영역에서는 경험 개선을 위한 활용이 가능하다. 향후 AI 활용의 신뢰도를 높이기 위해 진정성 강화 및 투명성 제고 방안을 지속적으로 모색해야 한다.
<--- Example End --->
"""
    return prompt_template.format(context=context_json_string, query=original_query)
