from src.composition_tables.baseclass import SemiGroup
from dataclasses import dataclass
import random

@dataclass
class RCC8(SemiGroup):
   name = 'rcc8'
   DC = 1
   EC = 2
   PO = 4
   TPP = 8
   NTPP = 16
   TPPI = 32
   NTPPI = 64
   EQ = 128

   str2int = {
      'DC':DC, 
      'EC':EC, 
      'PO':PO, 
      'TPP':TPP, 
      'NTPP':NTPP, 
      'TPPI':TPPI, 
      'NTPPI':NTPPI, 
      'EQ':EQ
      }
   
   elements = [DC , EC , PO , TPP , NTPP , TPPI , NTPPI , EQ]

   composition_table = {
      (DC, DC): [],
      (DC, EC): [DC,EC,PO,TPP,NTPP],
      (DC, PO): [DC,EC,PO,TPP,NTPP],
      (DC, TPP): [DC,EC,PO,TPP,NTPP],
      (DC, NTPP): [DC,EC,PO,TPP,NTPP],
      (DC, TPPI): [DC],
      (DC, NTPPI): [DC],
      (DC, EQ): [DC],
      
      (EC, DC): [DC,EC,PO,TPPI,NTPPI],
      (EC, EC): [DC,EC,PO,TPP,TPPI,EQ],
      (EC, PO): [DC,EC,PO,TPP,NTPP],
      (EC, TPP): [EC,PO,TPP,NTPP],
      (EC, NTPP): [PO,TPP,NTPP],
      (EC, TPPI): [DC,EC],
      (EC, NTPPI): [DC],
      (EC, EQ): [EC],

      (PO, DC): [DC,EC,PO,TPPI,NTPPI],
      (PO, EC): [DC,EC,PO,TPPI,NTPPI],
      (PO, PO): [],
      (PO, TPP): [PO,TPP,NTPP],
      (PO, NTPP): [PO,TPP,NTPP],
      (PO, TPPI): [DC,EC,PO,TPPI,NTPPI],
      (PO, NTPPI): [DC,EC,PO,TPPI,NTPPI],
      (PO, EQ): [PO],

      (TPP, DC): [DC],
      (TPP, EC): [DC,EC],
      (TPP, PO): [DC,EC,PO,TPP,NTPP],
      (TPP, TPP): [TPP,NTPP],
      (TPP, NTPP): [NTPP],
      (TPP, TPPI): [DC,EC,PO,TPP,TPPI,EQ],
      (TPP, NTPPI): [DC,EC,PO,TPPI,NTPPI],
      (TPP, EQ): [TPP],

      (NTPP, DC): [DC],
      (NTPP, EC): [DC],
      (NTPP, PO): [DC,EC,PO,TPP,NTPP],
      (NTPP, TPP): [NTPP],
      (NTPP, NTPP): [NTPP],
      (NTPP, TPPI): [DC,EC,PO,TPP,NTPP],
      (NTPP, NTPPI): [],
      (NTPP, EQ): [NTPP],

      (TPPI, DC): [DC,EC,PO,TPPI,NTPPI],
      (TPPI, EC): [EC,PO,TPPI,NTPPI],
      (TPPI, PO): [PO,TPPI,NTPPI],
      (TPPI, TPP): [PO,TPP,TPPI,EQ],
      (TPPI, NTPP): [PO,TPP,NTPP],
      (TPPI, TPPI): [TPPI,NTPPI],
      (TPPI, NTPPI): [NTPPI],
      (TPPI, EQ): [TPPI],

      (NTPPI, DC): [DC,EC,PO,TPPI,NTPPI],
      (NTPPI, EC): [PO,TPPI,NTPPI],
      (NTPPI, PO): [PO,TPPI,NTPPI],
      (NTPPI, TPP): [PO,TPPI,NTPPI],
      (NTPPI, NTPP): [PO,TPP,NTPP,TPPI,NTPPI,EQ],
      (NTPPI, TPPI): [NTPPI],
      (NTPPI, NTPPI): [NTPPI],
      (NTPPI, EQ): [NTPPI],

      (EQ, DC): [DC],
      (EQ, EC): [EC],
      (EQ, PO): [PO],
      (EQ, TPP): [TPP],
      (EQ, NTPP): [NTPP],
      (EQ, TPPI): [TPPI],
      (EQ, NTPPI): [NTPPI],
      (EQ, EQ): [EQ],


   }
   @classmethod
   def translate(self, BR):
      if BR == self.DC:
         return 'DC'
      if BR == self.EC:
         return 'EC'
      if BR == self.PO:
         return 'PO'
      if BR == self.TPP:
         return 'TPP'
      if BR == self.TPPI:
         return 'TPPI'
      if BR == self.NTPP:
         return 'NTPP'
      if BR == self.NTPPI:
         return 'NTPPI'
      if BR == self.EQ:
         return 'EQ'
   
   @classmethod
   def makeconsistent(self, csp,node_num):
      "relation composition sets need to be 8-bit floats"
      s = [random.randint(1,71) for i in range(node_num)]
      e = [s[i] + random.randint(1,17) for i in range(node_num)]

      for i in csp:
         for j in csp[i]:
            if i == j: csp[i][j] |= self.EQ
            if e[i] < s[j] or e[j] < s[i]: csp[i][j] |= self.DC
            elif e[i] == s[j] or e[j] == s[i]:  csp[i][j] |= self.EC
            elif s[i] >  s[j] and e[i] <  e[j]: csp[i][j] |= self.NTPP
            elif s[i] <  s[j] and e[i] >  e[j]: csp[i][j] |= self.NTPPI
            elif s[i] == s[j] and e[i] <  e[j]: csp[i][j] |= self.TPP
            elif s[i] == s[j] and e[i] >  e[j]: csp[i][j] |= self.TPPI
            elif s[i] >  s[j] and e[i] == e[j]: csp[i][j] |= self.TPP
            elif s[i] <  s[j] and e[i] == e[j]: csp[i][j] |= self.TPPI
            elif s[i] == s[j] and e[i] == e[j]: csp[i][j] |= self.EQ
            elif s[i] <  s[j] and e[i] <  e[j] and e[i] >  s[j]: csp[i][j] |= self.PO
            elif s[i] >  s[j] and e[i] >  e[j] and s[i] <  e[j]: csp[i][j] |= self.PO
            else: print ("ERROR")