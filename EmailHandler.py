'''
Created on Dec 27, 2017

@author: daniel
'''


import smtplib 
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

class EmailHandler:
    addr = "mrimathnotifier@gmail.com"
    addressBook = {'Danny': 'danielenricocahall@gmail.com', 
         'Dr.Bouaynaya': 'bouaynaya@rowan.edu', 
         'Oliver': 'palumb48@students.rowan.edu',
         'Dimah': "derad6@rowan.edu",
         'Alena': 'alenagusevarus@gmail.com',
         'Hassan': 'hfshaykh@uabmc.edu'}
 
    
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
        self.msg['To'] = self.addressBook.get(recipient)
        self.server.sendmail(self.addr, self.addressBook.get(recipient), self.msg.as_string())
    
    def finish(self):
        self.server.quit()
