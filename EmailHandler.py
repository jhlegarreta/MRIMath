'''
Created on Dec 27, 2017

@author: daniel
'''


import smtplib 
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

class EmailHandler:
    addr = "mrimathnotifier@gmail.com"
 
    
    def __init__(self):
        self.server = smtplib.SMTP('smtp.gmail.com', 587)
        self.server.ehlo()
        self.server.starttls()
        self.server.ehlo()
        self.server.login(self.addr, "mrimathpw")
        self.msg = MIMEMultipart()

        self.msg['From']  = self.addr
    
    def prepareMessage(self, subject, body):
        self.msg['Subject'] = subject
        self.msg.attach(MIMEText(body, 'plain'))
        
    def sendMessage(self, recipient):
        self.msg['To'] = recipient
        self.server.sendmail(self.addr, recipient, self.msg.as_string())
        self.server.quit()
