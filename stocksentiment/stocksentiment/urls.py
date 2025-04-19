"""
URL configuration for stocksentiment project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from core import views as cv

urlpatterns = [
    path('admin/', admin.site.urls),
    path('sentiment-analysis/', cv.sentiment_analysis_manual, name='sentiment_analysis'),
    path('predict_all_stock_prices/', cv.predict_all_stock_prices, name='predict_all_stock_prices'),
    path('get_predicted_stock_price/<str:company_name>/', cv.get_predicted_stock_price, name='get_predicted_stock_price'),
    path('api/stock-chart/', cv.stock_chart_api, name='stock_chart_api'),
    path('stock-chart/', cv.stock_chart_view, name='stock_chart_view'),
    path('api/companies/', cv.company_list, name='company_list'),
    path('api/reddit-posts/', cv.reddit_post_fetcher_by_company, name='reddit-posts'),
    path('api/company-poll/<str:company_name>/', cv.company_poll_api), # type: ignore
    path('api/news/all/', cv.all_company_news, name='all_company_news'),
    path('api/signup/', cv.signup_view, name='signup'),
    path('api/login/', cv.login_view, name='login'),
    path('api/search/', cv.search, name='search'), # type: ignore
    path('api/favourites/toggle/', cv.toggle_favourite, name='toggle_favourite'),
    path('api/favourites/', cv.get_favourites, name='get_favourites'),
]
