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
         'Dr. Hassan': 'hfshaykh@uabmc.edu'}
    body = ""
    
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
        self.body = body
        
    def sendMessage(self, recipient):
        self.msg['To'] = self.addressBook.get(recipient)
        self.body = "Hello " + self.addressBook.get(recipient) + "\n\n" + self.body
        self.body = self.body + "\n\n" + "Regards,\n Mr. HPC"
        self.msg.attach(MIMEText(self.body, 'plain'))
        self.server.sendmail(self.addr, self.addressBook.get(recipient), self.msg.as_string())
    
    def finish(self):
        self.server.quit()
