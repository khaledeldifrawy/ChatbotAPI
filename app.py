from flask import Flask, request
import numpy as np
import pickle
import json
from keras.models import load_model
from nltk_utils import tokenize, stem, bag_of_ward
from train import all_word,tags,xy,set_state

app = Flask(__name__)

model = load_model("model.h5")
intents = json.loads(open("new-dataset.json").read())


depreesion = " \nfollow this steps and all thing will be better :\n\n1. Practice Self-Care: Taking care of your physical and emotional needs is important for managing depression. This can include getting enough sleep, eating a balanced diet, exercising regularly, and engaging in activities that bring you joy. \n\n2. Seek professional help: Depression is a serious condition that may require medical attention. It is important to speak with a mental health professional who can provide guidance and support in managing your symptoms.\n\n3.Follow your treatment plan: If you are prescribed a medication, be sure to take it as directed by your doctor. Attend all therapy sessions and actively participate in your treatment plan.\n\n4.Challenging negative thoughts: Depression can often involve negative thoughts and self-criticism. Practice challenging these thoughts by looking for evidence that supports or contradicts them, and rephrasing them in a more positive and realistic light.\n\n5.Stay connected: Social support is important for managing depression. Make an effort to stay in touch with friends and loved ones, even if it feels challenging.\n\n6.Practicing relaxation techniques: Relaxation techniques such as deep breathing, meditation, or yoga can help manage stress and reduce symptoms of depression.\n\nSend a message if there are any inquiries regarding these instructions or if there is difficulty in applying them.\n"
Anxiety = " \nfollow this steps and all thing will be better :\n\n1.Identify triggers: Work with your therapist to identify situations, people, or events that trigger your anxiety. This can help you prepare and develop coping strategies.\n\n2.Gradual exposure: Gradual exposure to feared situations or triggers can be an effective way to manage anxiety. Work with a therapist to develop a gradual exposure plan that feels safe and manageable for you.\n\n3.Cognitive behavioral therapy (CBT): is one of the most effective forms of psychotherapy for anxiety disorders. In general, short-term cognitive behavioral therapy focuses on teaching specific skills to improve symptoms and gradually returning to activities the person avoids because of anxiety.\n\n4.Use positive self-talk: Replace negative self-talk with positive, supportive statements. This can help reduce anxiety and increase self-confidence.\n\n5.Avoid procrastination: Procrastination can increase anxiety by creating a sense of urgency and stress. Try to complete tasks as they come up, and break larger tasks into smaller, more manageable steps.\n\n6.Learn coping skills: Your therapist can teach you specific coping skills, such as cognitive-behavioral techniques or exposure therapy, to help manage your anxiety.\n\n7.Seek professional help: Anxiety disorders can be debilitating and it's important to speak with a mental health professional who can provide guidance and support in managing your symptoms.\n\n8.Practice relaxation techniques: Relaxation techniques like deep breathing, meditation, or yoga can help manage stress and reduce symptoms of anxiety.\n\n9.Challenge negative thoughts: Anxiety can often involve negative and catastrophic thinking. Practice challenging these thoughts by looking for evidence that supports or contradicts them, and reframing them in a more positive and realistic light.\n\n10.Maintain a healthy lifestyle: Taking care of your physical and emotional needs is important for managing anxiety. This can include getting enough sleep, eating a balanced diet, and exercising regularly.\n\nSend a message if there are any inquiries regarding these instructions or if there is difficulty in applying them.\n"
Addictive = " \nfollow this steps and all thing will be better :\n\n1.Create a support system: Building a support system of family, friends, or support groups can be helpful in managing addiction. Reach out to loved ones and consider joining a support group such as Alcoholics Anonymous or Narcotics Anonymous.\n\n2.Create a recovery plan: Work with your therapist or support group to create a recovery plan that includes goals, strategies, and a support system.\n\n3.Setting boundaries: Setting boundaries with people or situations that enable or lead to addiction can be helpful in managing addiction. Clearly articulate your boundaries and stick to them.\n\n4.Be accountable: Accountability can be helpful in managing addiction. Consider sharing your recovery plan with a trusted friend or family member and asking them to check in with you regularly.\n\n5.Celebrate small victories: Addiction recovery is a long and difficult process. Celebrate the small victories along the way, like completing a week of sobriety or reaching a milestone in your recovery plan.\n\n6.Attending support group meetings: Support group meetings, such as Alcoholics Anonymous or Narcotics Anonymous, can provide a safe and supportive space to share experiences and learn from others who have gone through similar struggles.\n\n7.Motivational Enhancement Therapy (MET): Motivational therapy is an approach that helps increase people's willingness to change. It can be useful for improving adherence and motivation to start and stay in treatment.\n\n8.Seek professional help: Addiction is a complex condition that requires professional help. Consider seeing a mental health professional who specializes in addiction treatment.\n\n9.Avoidance of triggers: Triggers such as stress, certain people or places, and emotional distress can lead to a relapse. Avoid triggers as much as possible and develop strategies to manage them as they arise.\n\n10.Learning new coping strategies: Addiction often involves the use of substances as a coping mechanism for stress or difficult feelings. Learn new coping strategies such as mindfulness, exercise, or hobbies to manage stress and emotions in a healthy way.\n\n11.Practice Self-Care: Taking care of your own physical and emotional needs is important for managing addiction. This can include getting enough sleep, eating a balanced diet, exercising regularly, and engaging in activities that bring you joy.\n\nSend a message if there are any inquiries regarding these instructions or if there is difficulty in applying them.\n"
PTSD = " \nfollow this steps and all thing will be better :\n\n1.Seek professional help: Postpartum depression or anxiety is a medical condition that requires professional treatment. A mental health professional or health care provider can provide guidance and support in managing your condition.\n\n2.Take care of your physical health: Taking care of your physical health can help improve your mental health. This can include getting enough sleep, eating a healthy diet, and getting regular physical activity.\n\n3.Ask for help: It is important to ask for help when you need it. Don't be afraid to reach out to friends, family, or other sources of support to help with tasks like cooking, cleaning, or babysitting.\n\n4.Attending support group meetings: Attending support group meetings can provide new moms with a safe and supportive space to share experiences and learn from others who are going through similar challenges.\n\n5.Make time for self-care: It's important to prioritize self-care, even if it feels challenging with a new baby. This can include taking a shower, reading a book, or engaging in other activities that bring you joy and relaxation.\n\n6.Challenge negative thoughts: When you notice negative thoughts, practice challenging them by looking for evidence to support or contradict them, and rephrase them in a more positive and realistic light.\n\n7.Gradually increase social support: Social support is important for managing postpartum depression or anxiety. Gradually increasing social activities, such as meeting friends or joining a moms' group, can help reduce feelings of isolation.\n\n8.Take it one day at a time: Recovering from postpartum depression or anxiety is a process, and it's important to take it one day at a time. Focus on small, achievable goals and celebrate your progress along the way.\n\nSend a message if there are any inquiries regarding these instructions or if there is difficulty in applying them.\n"
education_Disorder = " \nfollow this steps and all thing will be better :\n\n1.Diagnosis and Assessment: It's crucial to seek a proper diagnosis from a healthcare professional specializing in learning disorders. They will conduct assessments to identify specific areas of difficulty and provide recommendations for treatment and support.\n\n2.Individualized Education Plan (IEP): Work with educators, school administrators, and learning specialists to develop an individualized education plan (IEP) for the individual. The IEP outlines specific accommodations, modifications, and support services needed to address the learning disorder in an educational setting.\n\n3.Specialized Interventions: Engage in specialized interventions designed to address the specific challenges associated with the learning disorder.\n\n4.Assistive Technology: Explore and utilize assistive technology tools and resources that can aid in learning and compensate for specific difficulties. These may include text-to-speech software, speech recognition software, graphic organizers, or electronic organizers.\n\n5.Lifestyle Modifications: Adopting healthy lifestyle habits can support overall well-being and improve cognitive functioning. Encourage regular exercise, adequate sleep, a balanced diet, and stress management techniques.\n\n6.Parental and Teacher Involvement: Collaborate closely with parents, teachers, and support staff to ensure consistent communication and support. Regularly update them on progress, challenges, and any adjustments needed to the learning plan.\n\n7.Self-Advocacy Skills: Help the individual develop self-advocacy skills so they can communicate their needs, seek appropriate accommodations, and actively participate in their own learning process.\n\n8.Emotional Support: Provide a supportive and nurturing environment that acknowledges the individual's efforts and celebrates their progress. Encourage open communication and seek additional emotional support through counseling or support groups if necessary.\n\nSend a message if there are any inquiries regarding these instructions or if there is difficulty in applying them.\n"
ADHD = " \nfollow this steps and all thing will be better :\n\n1.Professional Evaluation and Diagnosis: Seek a comprehensive evaluation from a qualified healthcare professional, such as a psychiatrist or psychologist, to obtain an accurate diagnosis of ADHD. This will help guide the treatment and management strategies.\n\n2.Medication Management: Consult with a psychiatrist or healthcare provider experienced in treating ADHD to discuss medication options. Medication can help manage symptoms and improve focus and impulse control. Work closely with the healthcare professional to find the right medication and dosage that suits the individual's needs.\n3.Behavioral Therapy: Engage in behavioral therapy, specifically Cognitive-Behavioral Therapy (CBT) or ADHD-specific behavioral interventions. These therapies can help individuals develop coping strategies, organizational skills, time management techniques, and improve executive functioning.\n\n4.Education and Understanding: Learn about ADHD, its symptoms, and its impact on daily life. Knowledge about the disorder can help individuals understand their challenges, reduce self-blame, and develop strategies for self-management.\n\n5.Structure and Routine: Establish structured routines and schedules to provide a sense of stability and predictability. Use visual aids, calendars, planners, or smartphone apps to help organize tasks, deadlines, and responsibilities.\n\n6.Time Management Techniques: Break down tasks into smaller, manageable steps and set specific time limits for each step. Use timers or alarms to stay on track and manage time effectively.\n\n7.Environmental Modifications: Create an environment that minimizes distractions. Remove unnecessary clutter, noise, or visual stimuli from the workspace. Consider using noise-cancelling headphones or white noise machines to improve focus.\n\n8.Assistive Tools and Technology: Utilize tools and technology to assist with organization, time management, and task completion. This may include digital planners, reminder apps, productivity apps, or assistive technologies specifically designed for individuals with ADHD.\n\n9.Exercise and Physical Activity: Engage in regular physical exercise, as it can help reduce hyperactivity, improve focus, and enhance overall well-being. Choose activities that the individual enjoys, such as sports, swimming, or dance.\n\n10.Support System: Establish a strong support system that includes family, friends, teachers, or support groups. Surrounding oneself with understanding and supportive individuals can provide encouragement, guidance, and a sense of community.\n\n11.Self-Care: Encourage self-care practices, such as getting enough sleep, maintaining a healthy diet, and engaging in activities that promote relaxation and stress reduction. Adequate self-care can positively impact overall well-being and help manage ADHD symptoms.\n\nSend a message if there are any inquiries regarding these instructions or if there is difficulty in applying them.\n"
schizophrenic = " \nfollow this steps and all thing will be better :\n\n1.Therapy and Psychosocial Interventions: Engage in therapy, such as cognitive-behavioral therapy (CBT) or family therapy, to develop coping mechanisms, manage stress, improve communication, and enhance social skills.These interventions can also help with setting realistic goals, addressing self-esteem issues, and managing the impact of the disorder on daily life.\n\n2.Supportive Services: Seek support from mental health professionals, community organizations, and support groups specializing in schizophrenia. These resources can provide education, guidance, and a sense of community.\n\n3.Establish a Stable Routine: Create a structured daily routine that includes regular sleeping patterns, meal times, exercise, and engaging in meaningful activities. Maintaining a stable routine can help minimize stress, promote a sense of control, and manage symptoms.\n\n4.Healthy Lifestyle: Focus on adopting a healthy lifestyle. This includes regular exercise, a balanced diet, adequate sleep, and avoiding substance abuse, as substances can worsen symptoms and interfere with medication effectiveness.\n\n5.Stress Management: Learn stress management techniques, such as deep breathing exercises, mindfulness, relaxation techniques, or engaging in hobbies or activities that promote relaxation and stress reduction. Developing effective coping mechanisms can help manage stressors associated with schizophrenia.\n\n6.Social Support: Foster healthy social connections and maintain a support network. Engage in social activities, join support groups, and maintain open communication with family and friends. Social support can provide a sense of belonging, reduce isolation, and offer understanding and encouragement.\n\n7.Monitor Symptoms and Early Warning Signs: Learn to recognize early warning signs and symptoms of a potential relapse. Work with a mental health professional to develop a relapse prevention plan that outlines steps to take when symptoms escalate.\n\n8.Vocational and Educational Support: Seek vocational and educational support services that can assist in achieving employment or educational goals.These services can provide guidance, job training, or accommodations in educational settings.\n\n9.Self-Advocacy: Learn to advocate for oneself in healthcare settings, educational institutions, and workplaces. Develop effective communication skills to express needs and concerns, and ensure access to appropriate accommodations and support.\n\n10.Family Involvement: Encourage family members to participate in family therapy or support groups to better understand schizophrenia and develop strategies for supporting the individual. Family support can play a crucial role in the overall management of the disorder.\n\nSend a message if there are any inquiries regarding these instructions or if there is difficulty in applying them.\n"
Postpartum = " \nfollow this steps and all thing will be better :\n\n1.Seek professional help: Postpartum depression or anxiety is a medical condition that requires professional treatment. A mental health professional or health care provider can provide guidance and support in managing your condition.\n\n2.Take care of your physical health: Taking care of your physical health can help improve your mental health. This can include getting enough sleep, eating a healthy diet, and getting regular physical activity.\n\n3.Ask for help: It is important to ask for help when you need it. Don't be afraid to reach out to friends, family, or other sources of support to help with tasks like cooking, cleaning, or babysitting.\n\n4.Attending support group meetings: Attending support group meetings can provide new moms with a safe and supportive space to share experiences and learn from others who are going through similar challenges.\n\n5.Make time for self-care: It's important to prioritize self-care, even if it feels challenging with a new baby. This can include taking a shower, reading a book, or engaging in other activities that bring you joy and relaxation.\n\n6.Challenge negative thoughts: When you notice negative thoughts, practice challenging them by looking for evidence to support or contradict them, and rephrase them in a more positive and realistic light.\n\n7.Gradually increase social support: Social support is important for managing postpartum depression or anxiety. Gradually increasing social activities, such as meeting friends or joining a moms' group, can help reduce feelings of isolation.\n\n8.Take it one day at a time: Recovering from postpartum depression or anxiety is a process, and it's important to take it one day at a time. Focus on small, achievable goals and celebrate your progress along the way.\n\nSend a message if there are any inquiries regarding these instructions or if there is difficulty in applying them.\n"

def steps_print (Diagnosy,res):
    if Diagnosy == 'depreesion':
        temp = res+depreesion
        return temp
    elif Diagnosy == 'anxiety disorder':
        temp = res+Anxiety
        return temp
    elif Diagnosy == 'addictive':
        temp = res+Addictive
        return temp
    elif Diagnosy == 'Schizophrenia':
        temp = res+schizophrenic
        return temp
    elif Diagnosy == 'Postpartum':
        temp = res+Postpartum
        return temp
    elif Diagnosy == 'ADHD':
        temp = res+ADHD
        return temp       
    elif Diagnosy == 'PTSD':
        temp = res+PTSD
        return temp  
    elif Diagnosy == 'education':
        temp = res+education_Disorder
        return temp

@app.route("/chat", methods=["GET", "POST"])
def chatbot_response():
    flag = 0
    steps=0
    diagnosis = {
        "depression": 0,
        "anxiety disorder": 0,
        "addictive": 0,
        "Schizophrenia": 0,
        "Postpartum": 0,
        "ADHD": 0,
        "PTSD": 0,
        "education": 0,
    }

    msg = str(request.args.get('msg', ''))
    tokens = tokenize(msg)
    X = bag_of_ward(tokens, all_word)
    X = np.array(X)

    output = model.predict(np.array([X]))[0]
    predicted = np.argmax(output)
    tag = tags[predicted]

    if tag[-4:] == '_key':
        diagnosis[tag[:-4]] = 1

    prob = output[predicted]
    if prob > 0:
        if diagnosis.get(tag[:-3], -1) + 1 and diagnosis[tag[:-3]] == 0:
            diagnosis[tag[:-3]] = 1
            if tag[:-3] == 'addictive' and diagnosis['depression'] != 0:
                tag = tag
            else:
                tag = tag[:-3] + '_key'

        if diagnosis.get(tag[:-3], -1) + 1 and tag[-2] == 'E':
            flag = 1
            steps=1

        for intent in intents["intents"]:
            if tag == intent["tag"]:
                res = np.random.choice(intent['responses'])
                if steps == 1 :
                    res= steps_print(tag[:len(tag)-3],res)
                    steps = 0
             
        
    else:
        res = "Please start with a meaningful word."

    dict1 = {"cnt": res}
    return dict1

if __name__ == '__main__':
    app.run()
