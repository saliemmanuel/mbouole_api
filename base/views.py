from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler


@csrf_exempt
def predict(request):
    if request.method == 'POST':

        # Recupération des donnée avec le POST et Prétraitement des variables
        interestRate = robust_normalize_interest(request.POST["interestRate"])
        loanTerm = robust_normalize_loanterm(request.POST["loanTerm"])
        age = robust_normalize_age(request.POST["age"])
        loanAmount = robust_normalize_loanamount(request.POST["loanAmount"])
        sexe = normalize_sexe(request.POST["sexe"])
        revenu = robust_normalize_revenu(request.POST["revenu"])
        loanNbr = robust_normalize_loannbr(request.POST["loanNbr"])
        logement = robust_normalize_logement(request.POST["logement"])
        npaCharge = robust_normalize_npacharge(request.POST["npaCharge"])
        activiteSecondaire = robust_normalize_activite(
            request.POST["activiteSecondaire"])
        # Chargement du Models
        loaded_model = joblib.load("base/models/ModeleOctroifinal.joblib")
        # Prédiction du Models
        data = Octroi(loaded_model, interestRate,
                          loanTerm, age,
                          loanAmount, sexe, revenu, loanNbr,
                          logement, npaCharge, activiteSecondaire)
        # Récupération du résultat de prédiction
        response = {'prediction': data.tolist()}
        return JsonResponse(response)

    return JsonResponse({'error': 'Invalid request method.'})



def normalize_sexe(x):
   if x == "Femme":
     x = 0
   else:
     x = 1
   return x

def robust_normalize_interest(x):
        return np.divide(np.subtract(float(x), 18), 3)


def robust_normalize_loanterm(x):
        return np.divide(np.subtract(float(x), 12), 35)

def robust_normalize_age(x):
        return np.divide(np.subtract(float(x), 44), 16)

def robust_normalize_loannbr(x):
        return np.divide(np.subtract(float(x), 2), 3)

def robust_normalize_npacharge(x):
        return np.divide(np.subtract(float(x), 3), 3)

def robust_normalize_loanamount(x):
        return np.divide(np.subtract(float(x), 350000), 395000)

def robust_normalize_revenu(x):
        return np.divide(np.subtract(float(x), 149112), 117493)

def robust_normalize_activite(x):
  if x == "oui":
    x = 1
  else:
     x = 2
  return np.divide(np.subtract(int(x), 2), 1)

def robust_normalize_logement(x):
  if x == "Locataire":
    x = 1
  else:
     x = 2
  return np.divide(np.subtract(int(x), 1), 1)


def Octroi(ModeleOctroifinal, InterestRate, LoanTerm, age, LoanAmount, Sexe, Revenu, LoanNbr, Logement, NPAcharge, Activite_Secondaire):
    #c'est dans ce bloc que tu vas appliquer laa normalisation pour toutes les var sauf sexe
    x = np.array([InterestRate, LoanTerm, age, LoanAmount, Sexe, Revenu, LoanNbr, Logement, NPAcharge, Activite_Secondaire]).reshape(1,10)
    return ModeleOctroifinal.predict(x)

