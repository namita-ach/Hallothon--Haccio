#!/usr/bin/env python
# coding: utf-8

# In[4]:





# In[5]:


from flask import Flask,request,render_template,jsonify
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')



# In[6]:


model=pickle.load(open('pkmodel.pkl','rb'))


# In[8]:


@app.route('/')
def hello_world():
    return render_template('teacher_form.html')



# In[13]:


@app.route('/teacher_form', methods=['POST'])
def teacher_predict():
    try:
        topic = request.form['topic']
        mode_of_teaching = request.form['mode_of_teaching']
        
        input_data = {
            'topic': topic,
            'mode_of_teaching': mode_of_teaching
        }
        
        input_array = np.array([list(input_data.values())])
        probabilities = model.predict_proba(input_array)
        probability_positive_class = probabilities[0][1]
        output = '{0:.2f}'.format(probability_positive_class)
        return render_template('teacher_form.html',pred=output)
    except Exception as e:
        return jsonify({'error': str(e)})

        


# In[14]:


if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




