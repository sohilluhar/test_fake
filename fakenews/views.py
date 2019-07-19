from django.shortcuts import render
from django.http import HttpResponse
from . import act
from . import prediction
from . import main

def result(request):
	news=request.POST.get('news', None)
	print(news)
	act.a(news)
	if news!= None:
		print ("in if")
		ns=prediction.detecting_fake_news(news)
		a=main.analysis1()
		re=a.dd(news)

	else:
		ns=("","")
		news=""
		re=""
	print(ns[0])
	print(ns[1])
	return render(request,'fakenews/res.html',{'news':news,'ns1':ns[0],'ns2':ns[1],'re':re})

def home(req):

	return  render(req,'fakenews/home.html')

def res(req):
	news=req.POST.get('news', None)
	print(news)
	return news