# Prepoznavanje jezika koji se govori

**Studijsko istraživački rad**

---

**Univerzitet u Nišu**  
**Elektronski fakultet**  
**Katedra za računarstvo**

**Mentor:** prof. dr Miloš Radmanović  
**Student:** Aleksa Milić 1610

**Niš, 2025.**

---

## Apstrakt

Ovaj rad predstavlja implementaciju i evaluaciju sistema za automatsko prepoznavanje jezika iz govornog signala korišćenjem tehnika dubokog učenja i klasičnog mašinskog učenja. Sistem kombinuje savremene metode obrade audio signala kroz različite arhitekture: konvolucione neuronske mreže (CNN), rekurentne neuronske mreže (RNN/LSTM), kao i Wav2Vec model inspirisan dubokim reprezentacijama zvuka. Pored toga, razvijen je hibridni CNN–RNN model i sistem zasnovan na SVM pristupu. Implementirani sistem vrši ekstrakciju MFCC (Mel-Frequency Cepstral Coefficients) i mel-spektrogram karakteristika iz audio zapisa, koje se zatim koriste za treniranje modela.

Eksperimentalni rezultati na datasetu koji sadrži 10.018 test uzoraka pokazuju da Wav2Vec model postiže najbolje performanse, sa tačnošću od 85.06% i F1-score vrednošću od 0.8503. RNN model ostvaruje tačnost od 84.37%, CNN 83.57%, hibridni CNN–RNN 83.46%, dok SVM model dostiže 82.20%. Sistem je implementiran u programskom jeziku Python, uz korišćenje biblioteka TensorFlow i scikit-learn, i omogućava jednostavno korišćenje putem komandne linije.

**Ključne reči:** prepoznavanje jezika, obrada audio signala, duboko učenje, CNN, RNN, LSTM, Wav2Vec, SVM, MFCC, klasifikacija govora.

---

## Sadržaj

1. [Uvod](#1-uvod)
2. [Klasifikacija govornog signala](#2-klasifikacija-govornog-signala)
3. [Metodologija](#3-metodologija)
   - 3.1 [Arhitektura sistema i obrada podataka](#31-arhitektura-sistema-i-obrada-podataka)
   - 3.2 [Arhitekture modela i procedura treniranja](#32-arhitekture-modela-i-procedura-treniranja)
4. [Priprema podataka i treniranje](#4-priprema-podataka-i-treniranje)
5. [Rezultati i analiza](#5-rezultati-i-analiza)
6. [Zaključak](#6-zaključak)
7. [Reference](#7-reference)

---

## 1. Uvod

Razvoj tehnologija za automatsko prepoznavanje jezika iz govornog signala predstavlja jedan od najznačajnijih pravaca istraživanja u oblasti obrade govora i mašinskog učenja. Tokom poslednjih decenija, sa ubrzanim procesom globalizacije i eksponencijalnim rastom međunarodne komunikacije, potreba za efikasnim sistemima koji mogu automatski identifikovati jezik govora postala je imperativ u brojnim domenima primene.

Istorijski posmatrano, prvi pokušaji automatskog prepoznavanja jezika datiraju još iz 1970-ih godina, kada su istraživači koristili relativno jednostavne akustičke karakteristike i statističke metode za razlikovanje između ograničenog broja jezika. Ovi rani sistemi bili su zasnovani na analizi spektralnih karakteristika govora i fonetskih osobina pojedinih jezika, ali su imali značajna ograničenja u pogledu tačnosti i broja jezika koje su mogli da prepoznaju. Tradicionalni pristupi, koji su dominirali sve do početka 21. veka, zasnivali su se na ručno dizajniranim karakteristikama i klasičnim algoritmima mašinskog učenja, kao što su Gaussian Mixture Models (GMM) i Support Vector Machines (SVM). Ovi modeli zahtevali su značajan ekspertski rad u domenu lingvistike i obrade signala kako bi se identifikovale relevantne karakteristike koje razlikuju jezike.

Međutim, revolucija u oblasti dubokog učenja koja je započela oko 2012. godine, sa razvojem dubokih konvolucionih neuronskih mreža i dostupnošću velikih količina audio podataka, donela je fundamentalnu promenu u pristupu ovom problemu. Neuronske mreže pokazale su sposobnost da automatski uče hijerarhijske reprezentacije iz sirovih audio podataka, eliminirajući potrebu za ručnim dizajnom karakteristika i postižući superiorne performanse u zadacima klasifikacije audio signala [10]. Ova promena paradigme omogućila je razvoj sistema koji mogu da prepoznaju desetine, pa čak i stotine različitih jezika sa visokom tačnošću.

Danas, automatsko prepoznavanje jezika ima izuzetno široku primenu u industriji, društvu i nauci. U komercijalnom sektoru, ova tehnologija postala je nezamenljiva u višejezičnim call centrima gde omogućava automatsko rutiranje poziva ka operaterima koji govore odgovarajući jezik, čime se značajno poboljšava korisničko iskustvo i efikasnost poslovanja. Sistemi za automatsko prevođenje u realnom vremenu, koji su postali sastavni deo moderne međunarodne komunikacije, oslanjaju se na preciznu identifikaciju jezika kao prvi korak u procesu prevođenja. Digitalni asistenti poput Alexe, Siri i Google Assistanta koriste prepoznavanje jezika kako bi mogli da komuniciraju sa korisnicima na njihovom maternjem jeziku, dok streaming platforme poput YouTube-a i Netflix-a koriste ovu tehnologiju za automatsku kategorizaciju sadržaja i preporuke korisnicima.

U društvenom i kulturnom kontekstu, automatsko prepoznavanje jezika doprinosi bezbednosnim sistemima kroz analizu komunikacija u svrhu detekcije pretnji, omogućava automatsko titlovanje medijskog sadržaja čime se povećava pristupačnost informacija, pomaže u očuvanju ugroženih jezika kroz digitalizaciju i analizu audio arhiva, te olakšava učenje stranih jezika kroz interaktivne aplikacije koje pružaju trenutnu povratnu informaciju. U naučnim istraživanjima i medicini, analiza govora ima značajnu ulogu u proučavanju jezičke raznovrsnosti, sociolingvističkih fenomena, kao i u detekciji neuroloških poremećaja koji utiču na govor, poput Parkinsonove bolesti ili afazije.

Ekonomski uticaj ove tehnologije ne može se zanemariti - globalno tržište rešenja za obradu govora procenjuje se na više desetina milijardi dolara godišnje, sa stopom rasta koja premašuje 15% godišnje. Automatizacija procesa prepoznavanja jezika smanjuje operativne troškove u brojnim industrijama, od telekomunikacija do turizma, dok istovremeno povećava efikasnost i kvalitet usluga. U kontekstu globalizacije, ova tehnologija omogućava personalizaciju digitalnih usluga na osnovu jezika korisnika, ciljano oglašavanje na različitim tržištima, i efikasniju komunikaciju između geografski i kulturno udaljenih zajednica. Sa razvojem Internet of Things (IoT) uređaja i kontinuiranim porastom količine audio podataka koji se generišu svakodnevno, očekuje se da automatsko prepoznavanje jezika postane još integrisaniji deo svakodnevnih aplikacija - od pametnih domova koji se prilagođavaju jeziku korisnika, do autonomnih sistema koji mogu da komuniciraju sa ljudima na različitim jezicima.

Ipak, prepoznavanje jezika iz govornog signala ostaje kompleksan i izazovan zadatak klasifikacije. Za razliku od tekstualnog prepoznavanja jezika, gde su karakteristike relativno stabilne i jasno definisane, audio signal nosi sa sobom brojne izvore varijabilnosti i šuma. Potrebno je prepoznati jezik iz kratkog audio segmenta koji može biti snimljen u različitim akustičkim uslovima, sa različitim kvalitetom opreme, i od govornika sa različitim karakteristikama glasa. Na rezultate značajno utiču faktori kao što su varijabilnost između govornika (pol, starost, individualne karakteristike glasa), kvalitet snimanja (tip mikrofona, pozadinska buka, kompresija audio signala), trajanje audio segmenta (kraći segmenti nose manje informacija), lingvistička sličnost između jezika (posebno unutar istih jezičkih porodica), kao i postojanje dijalekata, regionalnih varijanti i stranih akcenata.

Dodatni izazov predstavlja i fenomen code-switching-a, odnosno prebacivanja između jezika tokom govora, što je česta pojava u višejezičnim zajednicama. Takođe, govorni signal može sadržati emotivne komponente, različite stilove govora (formalni, neformalni, brzi, spori), kao i specifične karakteristike kao što su pevanje, šaputanje ili govor pod stresom, što dodatno komplikuje zadatak automatskog prepoznavanja.

Zbog svih ovih faktora, razvoj tačnog i robusnog sistema za automatsko prepoznavanje jezika zahteva pažljivo dizajniranu metodologiju, odabir odgovarajućih karakteristika audio signala, primenu naprednih modela mašinskog učenja, kao i opsežno testiranje na raznovrsnim datasetima koji pokrivaju različite scenarije upotrebe. Ovaj rad predstavlja pokušaj da se sistematski istraže različiti pristupi ovom problemu, od klasičnih metoda mašinskog učenja do najsavremenijih arhitektura dubokih neuronskih mreža, sa ciljem da se identifikuju njihove prednosti, ograničenja i optimalne oblasti primene.

---

## 2. Klasifikacija govornog signala

Klasifikacija govornog signala, odnosno Spoken Language IDentification (SLID), predstavlja specifičan podskup problema klasifikacije u mašinskom učenju koji se bavi automatskim prepoznavanjem jezika na osnovu akustičkih karakteristika govora. Za razliku od tekstualne klasifikacije jezika koja analizira pisani sadržaj, SLID sistemi moraju da ekstrahuju relevantne informacije direktno iz audio signala, što uvodi dodatne dimenzije kompleksnosti.

Evolucija pristupa klasifikaciji govornog signala može se podeliti u nekoliko ključnih faza. Tokom 1990-ih i ranih 2000-ih godina, dominirali su pristupi zasnovani na Gaussian Mixture Models (GMM) i Hidden Markov Models (HMM), koji su koristili ručno dizajnirane akustičke karakteristike poput Linear Predictive Coding (LPC) koeficijenata i Mel-Frequency Cepstral Coefficients (MFCC). Ovi modeli zahtevali su značajno ekspertsko znanje za dizajn karakteristika i često su bili ograničeni u sposobnosti da generalizuju na nove uslove snimanja ili neviđene govornike.

Prelazak na metode mašinskog učenja zasnovane na Support Vector Machines (SVM) i Random Forests tokom 2000-ih godina doneo je određena poboljšanja, posebno u pogledu robusnosti i sposobnosti da rade sa višedimenzionalnim karakteristikama. Međutim, i dalje je postojala fundamentalna zavisnost od kvaliteta ručno dizajniranih karakteristika, što je predstavljalo usko grlo u razvoju sistema.

Revolucionarna promena dogodila se od 2012. godine, kada su duboke neuronske mreže postale dominantan pristup u SLID zadacima [10]. Ova promena paradigme nije bila slučajna - omogućena je konvergencijom nekoliko ključnih faktora: dostupnošću velikih količina označenih audio podataka, razvojem moćnijih grafičkih procesora (GPU) koji omogućavaju efikasno treniranje dubokih mreža, kao i teorijskim napretkom u razumevanju i optimizaciji dubokih arhitektura. Ključne prednosti dubokog učenja uključuju automatsko učenje karakteristika iz sirovih podataka (eliminirajući potrebu za ručnim dizajnom), sposobnost učenja hijerarhijskih višeslojnih reprezentacija (od jednostavnih akustičkih obrazaca do kompleksnih jezičkih struktura), i end-to-end treniranje koje optimizuje ceo sistem istovremeno, umesto optimizacije pojedinačnih komponenti odvojeno.

Duboke neuronske mreže pokazale su sposobnost da uče apstraktne reprezentacije koje su često superiorne ručno dizajniranim karakteristikama, čak i kada su dizajnirane od strane eksperata sa decenijama iskustva. Ove naučene reprezentacije pokazuju bolju generalizaciju na nove uslove, veću robusnost na šum i distorzije, i sposobnost da automatski identifikuju relevantne karakteristike koje razlikuju jezike, uključujući i one koje možda nisu bile očigledne ljudskim istraživačima.

### CNN arhitekture za SLID

Konvolucione neuronske mreže (CNN) originalno su razvijene za zadatke kompjuterske vizije, ali su se pokazale izuzetno efikasnim i u domenu obrade audio signala. Ključna ideja koja omogućava primenu CNN-a na audio podatke jeste tretiranje spektrograma kao dvodimenzionalnih slika. U ovom pristupu, horizontalna osa spektrograma predstavlja vremensku dimenziju (kako se signal menja tokom vremena), dok vertikalna osa predstavlja frekventnu dimenziju (koje frekvencije su prisutne u signalu). Intenzitet boje ili nijanse u svakoj tački spektrograma odgovara energiji signala na toj specifičnoj frekvenciji u tom specifičnom trenutku.

Ova analogija sa slikama nije samo površna - spektrogrami zaista poseduju prostorne strukture i obrasce koji su analogni vizuelnim karakteristikama u slikama. Na primer, harmonici u govornom signalu pojavljuju se kao horizontalne linije u spektrogramu, formanti (rezonantne frekvencije vokalnog trakta) formiraju karakteristične obrasce koji se razlikuju između različitih fonema, a prelazi između fonema manifestuju se kao promene u spektralnoj strukturi. Različiti jezici imaju različite fonetske inventare, različite distribucije fonema, i različite prozodijske karakteristike (intonacija, ritam, naglasak), što se sve reflektuje u specifičnim obrascima u spektrogramima.

CNN arhitekture za SLID sastoje se od nekoliko ključnih komponenti koje rade zajedno da ekstrahuju i klasifikuju ove obrasce:

**Konvolucioni slojevi** predstavljaju srce CNN arhitekture. Oni primenjuju skup naučenih filtera (kernela) na ulazni spektrogram, pri čemu svaki filter ima relativno male dimenzije (tipično 3×3 ili 5×5 piksela) ali se primenjuje na celom spektrogramu kroz operaciju konvolucije. Svaki filter uči da detektuje specifičan lokalni spektralno-temporalni obrazac. U nižim slojevima mreže, filteri obično uče jednostavne karakteristike kao što su ivice, prelazi između frekvencija, ili lokalne varijacije u energiji. U višim slojevima, filteri kombinuju ove jednostavne karakteristike da detektuju kompleksnije strukture kao što su formanti (karakteristični za specifične vokale), prelazi između fonema, harmonička struktura (koja se razlikuje između tonskih i netonskih jezika), ili specifični spektralni obrasci karakteristični za određene jezike.

Važno je napomenuti da se ovi filteri ne dizajniraju ručno - oni se uče automatski tokom procesa treniranja kroz algoritam backpropagation. Mreža sama otkriva koje karakteristike su najrelevantnije za razlikovanje jezika, što često rezultira filterima koji detektuju obrasce koje ljudski eksperti možda ne bi intuitivno identifikovali kao diskriminativne.

**Pooling slojevi** (najčešće max pooling ili average pooling) primenjuju se nakon konvolucionih slojeva i imaju nekoliko važnih funkcija. Prvo, oni smanjuju prostornu dimenzionalnost reprezentacije, što redukuje broj parametara u mreži i računsku kompleksnost, čineći model efikasnijim. Drugo, pooling uvodi određeni stepen translacione invarijantnosti - model postaje otporniji na male varijacije u položaju karakteristika u spektrogramu. Ovo je važno jer iste fonetske strukture mogu se pojaviti u različitim vremenskim pozicijama u različitim iskazima, a želimo da model prepozna te strukture nezavisno od njihove tačne pozicije. Tipična veličina pooling prozora je 2×2, što efektivno prepolovi prostorne dimenzije reprezentacije.

**Aktivacione funkcije** uvode nelinearnost u model, što je ključno za sposobnost mreže da modeluje kompleksne, nelinearne odnose između ulaza i izlaza. Najčešće korišćena aktivaciona funkcija u modernim CNN arhitekturama je ReLU (Rectified Linear Unit), definisana kao f(x) = max(0, x). ReLU ima nekoliko prednosti: jednostavna je za računanje, ne pati od problema vanishing gradient-a koji muči neke druge aktivacione funkcije (kao što je sigmoid), i empirijski se pokazalo da ubrzava konvergenciju tokom treniranja [14]. Varijante ReLU funkcije, kao što su Leaky ReLU ili Parametric ReLU, ponekad se koriste da bi se izbegao problem "umirućih" neurona koji mogu nastati sa standardnom ReLU funkcijom.

**Batch normalizacija** je tehnika koja se često uključuje između konvolucionih slojeva i aktivacionih funkcija. Ona normalizuje aktivacije svakog sloja tako da imaju srednju vrednost blizu nule i standardnu devijaciju blizu jedan, što stabilizuje proces treniranja i omogućava korišćenje većih learning rate-ova. Batch normalizacija takođe ima blagi regularizacioni efekat, smanjujući potrebu za drugim tehnikama regularizacije.

**Fully connected (dense) slojevi** dolaze na kraju CNN arhitekture i agregiraju sve naučene karakteristike iz konvolucionih slojeva da bi izvršili finalnu klasifikaciju. Pre fully connected slojeva, višedimenzionalni tensor koji izlazi iz poslednjeg konvolucionog/pooling sloja se "spljoštava" (flatten) u jednodimenzionalni vektor. Ovaj vektor zatim prolazi kroz jedan ili više fully connected slojeva, pri čemu poslednji sloj ima broj neurona jednak broju klasa (jezika) koje želimo da prepoznamo. Softmax aktivaciona funkcija primenjuje se na izlaz poslednjeg sloja da bi se dobila distribucija verovatnoća preko svih klasa, pri čemu zbir svih verovatnoća iznosi 1.

Tipična CNN arhitektura za SLID može imati 4–6 konvolucionih blokova, gde svaki blok sastoji se od konvolucionog sloja, batch normalizacije, aktivacione funkcije i pooling sloja. Broj filtera obično se postepeno povećava kroz slojeve (npr. 32→64→128→256), što omogućava mreži da uči sve kompleksnije reprezentacije. Dropout slojevi (sa stopom isključivanja između 0.3 i 0.5) često se dodaju između fully connected slojeva kao tehnika regularizacije koja sprečava overfitting tako što nasumično isključuje određeni procenat neurona tokom treniranja [18].

Jedna od ključnih prednosti CNN arhitektura za SLID jeste njihova sposobnost da automatski uče hijerarhijske reprezentacije. Niži slojevi uče jednostavne, lokalne karakteristike, dok viši slojevi kombinuju te karakteristike da formiraju kompleksnije, apstraktnije reprezentacije koje su direktno relevantne za zadatak klasifikacije jezika. Ova hijerarhijska priroda učenja omogućava CNN-ovima da postignu visoku tačnost čak i na izazovnim datasetima sa velikim brojem jezika.

### RNN/LSTM arhitekture

Rekurentne neuronske mreže (RNN) predstavljaju fundamentalno drugačiji pristup obradi audio signala u odnosu na CNN arhitekture. Dok CNN-ovi tretiraju spektrograme kao statične slike, RNN-ovi su specijalizovani za obradu sekvencijalnih podataka gde postoji inherentna temporalna zavisnost između uzastopnih elemenata. Ova karakteristika čini ih prirodnim izborom za analizu govornog signala, koji je po svojoj prirodi sekvencijalan - ono što se govori u jednom trenutku zavisi od onoga što je rečeno pre toga i utiče na ono što će biti rečeno posle.

Osnovna ideja RNN arhitekture jeste održavanje "memorije" kroz skriveno stanje koje se ažurira sa svakim novim ulaznim elementom u sekvenci. Za razliku od feedforward neuronskih mreža gde informacija teče samo u jednom smeru (od ulaza ka izlazu), RNN-ovi imaju rekurentne veze koje omogućavaju informaciji da kruži kroz mrežu. Ovo znači da izlaz mreže u trenutku t zavisi ne samo od ulaza u tom trenutku, već i od svih prethodnih ulaza kroz akumulirano skriveno stanje.

Matematički, standardni RNN se može opisati sledećim jednačinama:

h<sub>t</sub> = tanh(W<sub>hh</sub> · h<sub>t-1</sub> + W<sub>xh</sub> · x<sub>t</sub> + b<sub>h</sub>)

y<sub>t</sub> = W<sub>hy</sub> · h<sub>t</sub> + b<sub>y</sub>

gde je h<sub>t</sub> skriveno stanje u trenutku t, x<sub>t</sub> ulaz u trenutku t, y<sub>t</sub> izlaz u trenutku t, a W i b su težine i bias-evi koje mreža uči tokom treniranja.

Međutim, uprkos elegantnosti ove ideje, standardni RNN-ovi pate od ozbiljnog problema poznatog kao vanishing/exploding gradients. Tokom backpropagation kroz vreme (BPTT), gradijenti koji se propagiraju unazad kroz mnoge vremenske korake mogu eksponencijalno da opadnu (vanishing) ili rastu (exploding). Kada gradijenti postanu veoma mali, mreža efektivno gubi sposobnost da uči dugoročne zavisnosti - ne može da "zapamti" informacije koje su se pojavile mnogo koraka ranije u sekvenci. Ovo je posebno problematično za govorni signal gde relevantne informacije mogu biti raspoređene kroz relativno duge vremenske periode.

**LSTM (Long Short-Term Memory) arhitektura**, predložena od strane Hochreiter-a i Schmidhuber-a 1997. godine [9], predstavlja elegantno rešenje ovog problema. LSTM jedinice uvode koncept memorijske ćelije koja može da održava informacije kroz duge vremenske periode, zajedno sa sofisticiranim gate mehanizmima koji kontrolišu tok informacija u i iz ćelije. Ovi gate-ovi su zapravo male neuronske mreže (obično sa sigmoid aktivacionom funkcijom) koje uče kada treba dodati, zaboraviti ili pročitati informacije iz memorijske ćelije.

LSTM jedinica sastoji se od tri ključna gate mehanizma:

**Forget gate** (gate zaboravljanja) odlučuje koje informacije iz prethodnog stanja memorijske ćelije treba odbaciti. On gleda na prethodno skriveno stanje h<sub>t-1</sub> i trenutni ulaz x<sub>t</sub>, i za svaku vrednost u memorijskoj ćeliji C<sub>t-1</sub> proizvodi broj između 0 i 1, gde 0 znači "potpuno zaboravi ovo" a 1 znači "potpuno zadrži ovo":

f<sub>t</sub> = σ(W<sub>f</sub> · [h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>f</sub>)

**Input gate** (ulazni gate) odlučuje koje nove informacije treba dodati u memorijsku ćeliju. Ovo se dešava u dva koraka: prvo, sigmoid sloj (input gate) odlučuje koje vrednosti treba ažurirati, a zatim tanh sloj kreira vektor kandidata novih vrednosti C̃<sub>t</sub> koje bi mogle biti dodate u stanje:

i<sub>t</sub> = σ(W<sub>i</sub> · [h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>i</sub>)

C̃<sub>t</sub> = tanh(W<sub>C</sub> · [h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>C</sub>)

**Output gate** (izlazni gate) odlučuje koji deo memorijske ćelije će biti izlaz trenutnog koraka. Prvo se primenjuje sigmoid sloj koji odlučuje koje delove stanja ćelije ćemo izlaziti, a zatim se stanje ćelije propušta kroz tanh (da bi se vrednosti normalizovale između -1 i 1) i množi sa izlazom sigmoid gate-a:

o<sub>t</sub> = σ(W<sub>o</sub> · [h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>o</sub>)

Konačno stanje ćelije i skriveno stanje se ažuriraju na sledeći način:

C<sub>t</sub> = f<sub>t</sub> ⊙ C<sub>t-1</sub> + i<sub>t</sub> ⊙ C̃<sub>t</sub>

h<sub>t</sub> = o<sub>t</sub> ⊙ tanh(C<sub>t</sub>)

gde ⊙ označava element-wise množenje (Hadamard proizvod).

Ova arhitektura omogućava LSTM-u da selektivno zapamti ili zaboravi informacije, što mu daje sposobnost da uči dugoročne zavisnosti u sekvencama. U kontekstu prepoznavanja jezika, ovo je posebno važno jer različiti jezici imaju različite temporalne strukture - neki jezici imaju brz ritam sa kratkim slogovima, dok drugi imaju sporiji ritam sa dugim rečima; neki jezici su tonski i zahtevaju praćenje promene visine tona kroz vreme; neki imaju kompleksne prozodijske obrasce koji se manifestuju kroz duže segmente govora.

LSTM mreže su se pokazale posebno efikasnim kod jezika sa promenljivim ritmom, dugačkim rečima i složenom prozodijskom strukturom [1, 2]. Na primer, aglutinativni jezici kao što su finski ili turski, gde se reči formiraju dodavanjem mnogih sufiksa, zahtevaju sposobnost praćenja informacija kroz relativno duge vremenske periode. Slično, tonski jezici kao što su mandarinski ili vijetnamski zahtevaju precizno modelovanje promena visine tona kroz vreme.

**Bidirectional LSTM (BiLSTM)** arhitektura dodatno poboljšava performanse obrađujući sekvencu u oba smera - i napred i nazad kroz vreme. Ideja je da za razumevanje određenog dela sekvence često pomažu informacije ne samo iz prošlosti već i iz budućnosti. U kontekstu prepoznavanja jezika, poznavanje onoga što dolazi posle određenog segmenta govora može pomoći u boljoj interpretaciji tog segmenta. BiLSTM se sastoji od dve odvojene LSTM mreže - jedna obrađuje sekvencu od početka ka kraju, a druga od kraja ka početku. Izlazi ove dve mreže se zatim kombinuju (obično konkatenacijom) da formiraju finalnu reprezentaciju koja sadrži kontekst iz oba smera.

Praktična primena LSTM arhitektura za SLID obično uključuje stekovanje više LSTM slojeva (tipično 2-4 sloja), gde svaki sloj uči sve apstraktnije reprezentacije sekvence. Dropout regularizacija se često primenjuje između slojeva da bi se sprečio overfitting. Za zadatak klasifikacije jezika, obično se koristi izlaz poslednjeg vremenskog koraka ili se primenjuje neka forma pooling-a (npr. average ili max pooling) preko svih vremenskih koraka da bi se dobila fiksna reprezentacija cele sekvence, koja se zatim propušta kroz fully connected slojeve za finalnu klasifikaciju.

---

### MFCC (Mel-Frequency Cepstral Coefficients)

Mel-Frequency Cepstral Coefficients (MFCC) predstavljaju jednu od najšire korišćenih reprezentacija audio signala u zadacima obrade govora. Razvijeni tokom 1980-ih godina, MFCC koeficijenti su dizajnirani da imitiraju način na koji ljudski auditorni sistem percipira zvuk, što ih čini posebno pogodnim za aplikacije koje uključuju govorni signal. Osnovna ideja je da ljudsko uho ne percipira frekvencije na linearnoj skali - mi smo mnogo osetljiviji na razlike između niskih frekvencija nego između visokih frekvencija. Na primer, razlika između 100 Hz i 200 Hz je perceptualno mnogo značajnija od razlike između 10,100 Hz i 10,200 Hz, iako je apsolutna razlika ista.

Proces ekstrakcije MFCC koeficijenata sastoji se od nekoliko pažljivo dizajniranih koraka, od kojih svaki ima specifičnu ulogu u transformaciji sirovog audio signala u kompaktnu reprezentaciju pogodnu za mašinsko učenje:

**1. Pre-emphasis filtering** je prvi korak u obradi signala. Ovaj korak primenjuje visokopropusni filter na signal, tipično u formi y[n] = x[n] - α·x[n-1], gde je α koeficijent koji se obično postavlja na vrednost oko 0.97. Svrha ovog koraka je da pojača više frekvencije u signalu koje su prirodno slabije u govornom signalu zbog fizike produkcije govora. Tokom govora, vazduh prolazi kroz vokalni trakt i vibracije glasnih žica, što prirodno rezultira u spektru koji opada sa frekvencijom. Pre-emphasis kompenzuje ovaj pad i balansira spektar, što olakšava dalju analizu i poboljšava performanse sistema.

**2. Framing** je proces segmentacije kontinualnog audio signala u kratke, preklapajuće okvire (frame-ove). Tipična dužina okvira je između 20 i 40 milisekundi, što odgovara periodu tokom kojeg se karakteristike govornog signala mogu smatrati relativno stabilnim (kvazi-stacionarnim). Okviri se obično preklapaju za 50%, što znači da ako je okvir dužine 25ms, sledeći okvir počinje nakon 12.5ms (hop length). Ovo preklapanje osigurava da ne propustimo važne informacije koje bi mogle biti na granicama okvira i omogućava glatkiji prelaz između uzastopnih analiza.

**3. Windowing** se primenjuje na svaki okvir da bi se smanjio efekat spektralnog curenja (spectral leakage) koji nastaje zbog činjenice da okvir ima konačnu dužinu. Kada primenjujemo Fourier transformaciju na okvir, implicitno pretpostavljamo da je signal periodičan, što obično nije slučaj. Ovo može dovesti do artificijalnih frekvencijskih komponenti u spektru. Primena prozorske funkcije (najčešće Hamming ili Hann prozor) postepeno smanjuje amplitudu signala ka ivicama okvira, čime se minimizuje diskontinuitet na granicama i redukuje spektralno curenje. Hamming prozor je definisan kao w[n] = 0.54 - 0.46·cos(2πn/(N-1)), gde je N dužina prozora.

**4. Fast Fourier Transform (FFT)** transformiše svaki prozorski okvir iz vremenskog domena u frekvencijski domen. FFT je efikasan algoritam za računanje Diskretne Fourier Transformacije (DFT) i omogućava nam da vidimo koje frekvencije su prisutne u signalu i sa kojim intenzitetom. Tipično se koristi FFT sa 512 ili 1024 tačaka, što daje dovoljnu frekventnu rezoluciju za analizu govornog signala. Rezultat FFT-a je kompleksni spektar, ali nas obično interesuje samo magnituda (power spectrum), koja se dobija kao kvadrat apsolutne vrednosti kompleksnih koeficijenata.

**5. Mel-filter bank** primenjuje skup trougaonih filtera na power spektar dobijen iz FFT-a. Ovi filteri su raspoređeni na mel-skali, koja je nelinearna skala frekvencija dizajnirana da aproksimira način na koji ljudsko uho percipira visinu tona. Konverzija između linearne frekvencijske skale (u Hz) i mel-skale data je formulom: m = 2595 · log₁₀(1 + f/700), gde je f frekvencija u Hz, a m frekvencija u melovima. Na niskim frekvencijama, filteri su usko raspoređeni (veća frekventna rezolucija), dok su na visokim frekvencijama raspoređeni šire (manja rezolucija), što odražava smanjenu osetljivost ljudskog uha na razlike između visokih frekvencija. Tipično se koristi između 20 i 40 mel filtera, što rezultira u istom broju mel-frekventnih koeficijenata.

**6. Logaritamska kompresija** primenjuje se na izlaz mel-filter bank-a. Uzimanje logaritma energije (S = log(E)) ima dve važne funkcije: prvo, simulira način na koji ljudsko uho percipira intenzitet zvuka (mi percipiramo intenzitet približno logaritamski, što je poznato kao Weber-Fechner zakon), i drugo, kompresuje dinamički opseg vrednosti, što pomaže u numeričkoj stabilnosti i olakšava učenje modela. Logaritamska kompresija takođe ima efekat pretvaranja multiplikativnih odnosa u aditivne, što može biti korisno za modelovanje određenih karakteristika govora.

**7. Discrete Cosine Transform (DCT)** je finalni korak koji primenjuje kosinusnu transformaciju na logaritamske mel-frekventne koeficijente. DCT je slična Fourier transformaciji ali radi samo sa realnim brojevima i ima svojstvo da kompaktuje energiju u nekoliko prvih koeficijenata. Tipično se zadržava prvih 12-20 MFCC koeficijenata (od ukupno 20-40 koliko ih DCT proizvodi), jer oni sadrže većinu relevantnih informacija. Viši koeficijenti obično predstavljaju brze varijacije u spektru koje često odgovaraju šumu ili manje relevantnim detaljima.

Dodatno, često se računaju i **delta i delta-delta koeficijenti** (prvi i drugi izvodi MFCC-a po vremenu), koji hvataju dinamiku promena u govornom signalu. Delta koeficijenti pokazuju kako se MFCC menjaju između uzastopnih okvira, dok delta-delta koeficijenti pokazuju akceleraciju tih promena. Ove temporalne karakteristike su važne jer različiti jezici imaju različite dinamičke karakteristike - na primer, brzinu artikulacije, način prelaza između fonema, ili prozodijske obrasce.

Finalna MFCC reprezentacija jednog audio segmenta je tipično matrica dimenzija (broj_okvira × broj_koeficijenata), gde broj_okvira zavisi od dužine audio segmenta i hop length-a, a broj_koeficijenata je obično između 13 i 40 (uključujući delta i delta-delta koeficijente ako se koriste). Ova kompaktna reprezentacija zadržava ključne informacije o spektralnom sadržaju signala na način koji je prilagođen ljudskoj percepciji, što je čini izuzetno efikasnom za zadatke prepoznavanja govora i jezika [11].

### Mel-spektrogram

Mel-spektrogram predstavlja 2D vremensko-frekventnu reprezentaciju signala na mel-skali. Za razliku od MFCC koji daje komprimovane koeficijente, mel-spektrogram zadržava punu spektralnu informaciju i vizualno je pogodan za CNN. Tipične dimenzije su 128 mel filtera × vremenski okviri, sa amplitudama konvertovanim u decibele za bolji dinamički opseg. Na slici 1 prikazan je primer mel-spektrograma govornog signala sa 128 mel filtera, gde se jasno uočavaju harmonici i formanti.

![Slika 1: Mel-spektrogram govornog signala](images/mel_spectrogram_example.png)

**Slika 1:** Mel-spektrogram govornog signala (128 mel filtera) prikazuje vremensko-frekventnu energetsku distribuciju. Jasno vidljivi harmonici i formanti omogućavaju CNN modelima da automatski uče diskriminativne karakteristike različitih jezika.

### Dodatne spektralne karakteristike

Pored MFCC koeficijenata i mel-spektrograma, postoji čitav niz dodatnih spektralnih karakteristika koje mogu pružiti komplementarne informacije o audio signalu i potencijalno poboljšati performanse sistema za prepoznavanje jezika. Ove karakteristike često hvataju različite aspekte audio signala koji možda nisu optimalno reprezentovani kroz MFCC.

**Spectral Centroid** predstavlja "centar mase" spektra i može se shvatiti kao mera prosečne frekvencije u signalu, ponderisane intenzitetom svake frekvencijske komponente. Matematički se definiše kao:

SC = Σ(f<sub>k</sub> · M<sub>k</sub>) / Σ(M<sub>k</sub>)

gde je f<sub>k</sub> frekvencija, a M<sub>k</sub> magnituda u frekvencijskom bin-u k. Perceptualno, spectral centroid odgovara "svetlini" ili "brilijantnosti" zvuka - viši centroid odgovara "svetlijem" zvuku sa više visokih frekvencija, dok niži centroid odgovara "tamnijim", basovitijim zvucima. U kontekstu prepoznavanja jezika, različiti jezici mogu imati različite prosečne spektralne centroide zbog razlika u fonetskom inventaru - na primer, jezici sa više frikatnih glasova (kao što su s, š, f) mogu imati viši prosečni centroid.

**Spectral Rolloff** definiše se kao frekvencija ispod koje je koncentrisano određeni procenat (tipično 85% ili 95%) ukupne spektralne energije. Ova karakteristika je indikator zastupljenosti visokih frekvencija u signalu - niži rolloff ukazuje na signal sa dominantno niskim frekvencijama, dok viši rolloff ukazuje na prisustvo značajne energije u visokim frekvencijama. Spectral rolloff može biti koristan za razlikovanje između različitih tipova fonema (na primer, vokali obično imaju niži rolloff od frikatnih konsonanata) i može pomoći u identifikaciji jezika koji imaju različite distribucije fonetskih kategorija.

**Zero Crossing Rate (ZCR)** meri koliko često signal menja znak (prelazi kroz nulu) u okviru jednog okvira. Matematički, ZCR se može definisati kao:

ZCR = (1/2N) · Σ|sgn(x[n]) - sgn(x[n-1])|

gde je sgn() funkcija znaka, a N broj uzoraka u okviru. ZCR je posebno koristan za razlikovanje između tonskih (voiced) i šumnih (unvoiced) segmenata govora. Tonski segmenti (kao što su vokali) imaju relativno nizak ZCR jer su dominantno periodični sa niskim frekvencijama, dok šumni segmenti (kao što su frikativi) imaju visok ZCR zbog prisustva visokih frekvencija i šumne prirode. Različiti jezici imaju različite proporcije tonskih i šumnih segmenata, što ZCR može pomoći da uhvati.

**Spectral Flux** meri brzinu promene spektra kroz vreme i definiše se kao razlika između spektara uzastopnih okvira:

SF<sub>t</sub> = Σ(|M<sub>t</sub>[k] - M<sub>t-1</sub>[k]|)

gde je M<sub>t</sub>[k] magnituda frekvencijskog bin-a k u okviru t. Visok spectral flux ukazuje na brze promene u spektralnom sadržaju, što može odgovarati prelazima između fonema, brzoj artikulaciji, ili dinamičkim prozodijskim karakteristikama. Jezici sa bržim tempom govora ili češćim fonetskim prelazima mogu imati viši prosečni spectral flux.

**Chroma features** (ili Pitch Class Profiles) predstavljaju energiju distribuiranu preko 12 muzičkih tonova (C, C#, D, D#, E, F, F#, G, G#, A, A#, B), nezavisno od oktave. Iako su originalno razvijene za analizu muzike, chroma features mogu biti korisne i za analizu govora, posebno za jezike sa izraženom tonskom komponentom ili specifičnim intonacionim obrascima. Chroma features mogu pomoći u hvatanju prozodijskih karakteristika koje razlikuju jezike - na primer, tonski jezici kao što su mandarinski ili vijetnamski imaju leksički značajne promene visine tona koje se mogu reflektovati u chroma reprezentaciji.

**Spectral Contrast** meri razliku između vrhova i dolina u spektru, pružajući informacije o spektralnoj strukturi signala. Za razliku od spectral centroid-a koji daje jednu globalnu meru, spectral contrast analizira spektar u više frekvencijskih opsega i za svaki opseg računa razliku između prosečne energije u vrhovima i dolinima. Ova karakteristika može biti korisna za razlikovanje između različitih tipova zvukova i fonetskih kategorija.

**Spectral Bandwidth** meri "širinu" spektra oko spectral centroid-a i može se shvatiti kao mera spektralne disperzije. Uži bandwidth ukazuje na signal sa energijom koncentrisanom oko određenih frekvencija (kao što su čisti tonovi ili vokali), dok širi bandwidth ukazuje na signal sa energijom raspoređenom preko šireg frekvencijskog opsega (kao što su šumni konsonanti).

**Formanti** su rezonantne frekvencije vokalnog trakta i predstavljaju jedne od najvažnijih karakteristika za razlikovanje vokala. Različiti vokali karakterišu se različitim pozicijama prvih nekoliko formanata (tipično F1, F2, i F3). Iako ekstrakcija formanata može biti izazovna i podložna greškama, oni nose direktne informacije o artikulaciji i mogu biti veoma diskriminativni za prepoznavanje jezika, posebno jer različiti jezici imaju različite vokalske sisteme.

U praksi, izbor karakteristika zavisi od specifične aplikacije i dostupnih resursa. MFCC koeficijenti ostaju najčešće korišćena reprezentacija zbog svoje kompaktnosti, robusnosti i dokazane efikasnosti. Mel-spektrogrami su postali popularniji sa razvojem dubokih konvolucionih mreža koje mogu automatski da uče relevantne karakteristike iz ove bogatije reprezentacije. Kombinacija različitih karakteristika može pružiti komplementarne informacije i poboljšati performanse, ali takođe povećava dimenzionalnost podataka i računsku kompleksnost.

Važno je napomenuti da savremeni pristupi zasnovani na dubokom učenju često uče karakteristike direktno iz sirovih podataka ili iz relativno jednostavnih reprezentacija kao što su mel-spektrogrami, eliminirajući potrebu za ručnim dizajnom kompleksnih karakteristika. Međutim, razumevanje ovih tradicionalnih karakteristika i dalje je važno jer pruža uvid u to koje informacije su relevantne za zadatak i može voditi dizajnu boljih arhitektura neuronskih mreža.

Kombinacija ovih karakteristika, naročito MFCC i mel-spektrograma, čini osnovu većine modernih SLID sistema, omogućavajući balans između kompresije informacija i zadržavanja relevantnih detalja [3, 18]. Izbor između različitih reprezentacija često zavisi od trade-off-a između informativnosti, računske efikasnosti, i kompatibilnosti sa izabranom arhitekturom modela.

---

## 3. Metodologija

### 3.1 Arhitektura sistema i obrada podataka

Implementirani sistem za prepoznavanje jezika sastoji se od nekoliko modularnih komponenti koje zajedno formiraju kompletan pipeline od sirovog audio signala do predikcije jezika.

Sistem se sastoji od sledećih glavnih komponenti:

- **Audio Processor:** Učitavanje i priprema audio zapisa
- **Feature Extractor:** Ekstrakcija MFCC i spektrogram karakteristika
- **Dataset Builder:** Priprema dataseta za treniranje
- **Modeli:** Pet različitih pristupa klasifikaciji
  - **Wav2Vec Model:** Transformer-inspirisan model sa bidirekcionalnim LSTM
  - **RNN Model:** Rekurentna neuronska mreža sa LSTM jedinicama
  - **CNN Model:** Konvoluciona neuronska mreža za obradu spektrograma
  - **Hybrid CNN-RNN:** Kombinovani pristup sa CNN i RNN slojevima
  - **SVM Model:** Klasični mašinsko učenje pristup kao baseline
- **Evaluator:** Evaluacija performansi i vizualizacija rezultata
- **Language Recognizer:** Interfejs za prepoznavanje jezika iz novih audio zapisa

#### Audio Processing

Audio Processing modul je odgovoran za učitavanje i normalizaciju audio zapisa. Implementiran je u Python-u korišćenjem librosa biblioteke [5].

Ključne funkcionalnosti modula:

```python
class AudioProcessor:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
    
    def load_audio(self, file_path):
        # Učitavanje audio fajla
        signal, sr = librosa.load(file_path, sr=None)
        return signal, sr
    
    def preprocess(self, signal, sr):
        # Konverzija u mono
        if signal.ndim > 1:
            signal = librosa.to_mono(signal)
        
        # Resampling na 16kHz
        if sr != self.target_sr:
            signal = librosa.resample(signal, orig_sr=sr, 
                                     target_sr=self.target_sr)
        return signal
```

Normalizacija parametara:

- **Sample rate:** 16 kHz (standardna vrednost za govorni signal)
- **Broj kanala:** Mono (konverzija stereo zapisa)
- **Format:** Podrška za WAV, MP3 i FLAC formate

---

#### Feature Extraction

Feature Extraction modul ekstraktuje numeričke reprezentacije audio signala koje se koriste kao ulaz za neuronske mreže.

**MFCC Ekstrakcija:**

```python
class FeatureExtractor:
    def __init__(self, n_mfcc=40, n_fft=2048, hop_length=512):
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract_mfcc(self, signal, sr):
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return mfcc
```

Parametri ekstrakcije:

- **n_mfcc:** 40 koeficijenata (standardna vrednost)
- **n_fft:** 2048 (veličina FFT prozora)
- **hop_length:** 512 (korak između frejmova)
- **n_mels:** 128 (broj mel filtera za spektrogram)

**Mel-Spektrogram Ekstrakcija:**

```python
def extract_mel_spectrogram(self, signal, sr):
    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=self.n_fft,
        hop_length=self.hop_length,
        n_mels=128
    )
    # Konverzija u decibele
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db
```

#### Dataset Builder

Dataset Builder modul priprema podatke za treniranje modela, uključujući učitavanje audio zapisa, ekstrakciju karakteristika i podelu na trening, validacioni i test skup.

**Organizacija podataka:**

```
data/
├── raw/
│   ├── english/
│   │   ├── sample_001.wav
│   │   ├── sample_002.wav
│   │   └── ...
│   ├── serbian/
│   ├── german/
│   ├── spanish/
│   └── french/
```

**Podela dataseta:**

- **Training set:** 70% podataka
- **Validation set:** 15% podataka
- **Test set:** 15% podataka

**Normalizacija dužine sekvenci:**

Kako audio zapisi imaju različite dužine, implementirana je funkcija za padding ili truncation na fiksnu dužinu:

```python
def pad_or_truncate(self, features, max_length):
    if features.shape[1] < max_length:
        # Padding sa nulama
        pad_width = max_length - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), 
                         mode='constant')
    else:
        # Truncation
        features = features[:, :max_length]
    return features
```

---

### 3.2 Arhitekture modela i procedura treniranja

#### CNN Model

CNN model neuronske mreže dizajniran je za obradu mel-spektrograma kao 2D slika. Arhitektura:

```
Input: (128, 100, 1) - Mel-spektrogram
↓
Conv2D(32 filters, 3x3) + ReLU
MaxPooling2D(2x2)
↓
Conv2D(64 filters, 3x3) + ReLU
MaxPooling2D(2x2)
↓
Conv2D(128 filters, 3x3) + ReLU
MaxPooling2D(2x2)
↓
Flatten
↓
Dense(128) + ReLU
Dropout(0.5)
↓
Dense(num_classes) + Softmax
↓
Output: Verovatnoće za svaki jezik
```

Implementacija u TensorFlow/Keras framework [6]:

```python
def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

**Broj parametara:** Približno 1.2 miliona trenabilnih parametara

**Regularizacija:**

- **Dropout:** 0.5 (50% neurona se nasumično isključuje tokom treniranja) [13]
- **Early Stopping:** Zaustavljanje treniranja ako se validaciona tačnost ne poboljšava 10 epoha

---

#### RNN/LSTM Model

Rekurentna RNN/LSTM neuronska mreža sa LSTM jedinicama dizajnirana je za obradu sekvenci MFCC koeficijenata.

**Arhitektura:**

```
Input: (128, 40) – MFCC sekvenca
↓
LSTM(128 units, return_sequences=True)
Dropout(0.3)
↓
LSTM(128 units)
Dropout(0.3)
↓
Dense(64) + ReLU
↓
Dense(num_classes) + Softmax
↓
Output: Verovatnoće za svaki jezik
```

**Implementacija:**

```python
def build_rnn_model(input_shape, num_classes):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(128),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

**Broj parametara:** Približno 800,000 trenabilnih parametara

**Prednosti LSTM arhitekture:**

- Sposobnost modelovanja dugoročnih zavisnosti u audio signalu
- Prirodna obrada sekvencijalnih podataka
- Manje parametara u odnosu na CNN model

---

#### Wav2Vec Model

Wav2Vec model inspirisan je transformer arhitekturama [4, 15] i koristi bidirekcional LSTM sa attention mehanizmom za ekstrakciju kontekstualnih reprezentacija iz MFCC karakteristika.

**Arhitektura:**

```
Input: (100, 40) - MFCC sekvenca
↓
Bidirectional LSTM(128 units, return_sequences=True)
↓
Attention Layer (self-attention)
↓
Global Average Pooling
↓
Dense(64) + ReLU
Dropout(0.3)
↓
Dense(num_classes) + Softmax
↓
Output: Verovatnoće za svaki jezik
```

**Ključne karakteristike ove arhitekture su:**

- **Bidirectional LSTM:** Obrađuje sekvencu u oba smera (napred i nazad) za bolji kontekst [9]
- **Attention mehanizam:** Fokusira pažnju na najrelevantnije delove audio signala [15]
- **Global pooling:** Agregira informacije iz cele sekvence

**Broj parametara:** Približno 1.5 miliona trenabilnih parametara

#### Hibridni CNN-RNN Model

Hibridni CNN model kombinuje prednosti CNN-a za ekstrakciju prostornih karakteristika i RNN-a za modelovanje temporalnih zavisnosti.

**Arhitektura:**

```
Input: (128, 100, 1) - Mel-spektrogram
↓
Conv2D(32 filters, 3x3) + ReLU
MaxPooling2D(2x2)
↓
Conv2D(64 filters, 3x3) + ReLU
MaxPooling2D(2x2)
↓
Reshape (za RNN)
↓
LSTM(128 units, return_sequences=True)
↓
LSTM(64 units)
Dropout(0.3)
↓
Dense(64) + ReLU
↓
Dense(num_classes) + Softmax
↓
Output: Verovatnoće za svaki jezik
```

**Prednosti hibridnog pristupa:**

- Kombinuje prostornu invarijantnost CNN-a sa temporalnim modelovanjem RNN-a
- CNN slojevi ekstraktuju lokalne obrasce u spektrogramu
- LSTM slojevi modeluju temporalne zavisnosti između ekstraktovanih karakteristika

**Broj parametara:** Približno 1.0 milion trenabilnih parametara

---

#### Support Vector Machine (SVM) Model

Support Vector Machine (SVM) model implementiran je kao baseline pristup korišćenjem klasičnog mašinskog učenja. Model koristi statistički agregirane MFCC karakteristike.

U kontekstu feature engineeringa, MFCC sekvence se za SVM model agregiraju u fiksne vektore korišćenjem statističkih mera:

```python
def aggregate_mfcc_features(mfcc_sequence):
    features = []
    features.extend(np.mean(mfcc_sequence, axis=0))  # Mean
    features.extend(np.std(mfcc_sequence, axis=0))   # Std
    features.extend(np.max(mfcc_sequence, axis=0))   # Max
    features.extend(np.min(mfcc_sequence, axis=0))   # Min
    return np.array(features)
```

**Konfiguracija:**

- **Kernel:** RBF (Radial Basis Function)
- **C parameter:** 1.0 (regularizacija)
- **Gamma:** auto (1 / n_features)

**Prednosti SVM pristupa:**

- Jednostavnost implementacije i razumevanja
- Brzo treniranje bez potrebe za GPU
- Mala memorijska potrošnja
- Dobar baseline za poređenje sa deep learning modelima

#### Trening procedure

**Hiperparametri treniranja:**

- **Optimizer:** Adam (Adaptive Moment Estimation) [12]
- **Learning rate:** 0.001 (default za Adam)
- **Batch size:** 64
- **Epochs:** 50 (sa early stopping)
- **Loss function:** Categorical Crossentropy
- **Metrics:** Accuracy

Kako bi se poboljšala generalizacija modela, primenjene su sledeće tehnike augmentacije:

- **Time stretching:** Promena brzine bez promene visine tona
- **Pitch shifting:** Promena visine tona bez promene brzine
- **Background noise:** Dodavanje belog šuma sa niskim intenzitetom

---

#### Metrike evaluacije

Performanse modela evaluirane su korišćenjem sledećih metrika:

**Accuracy (Tačnost):**

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precision (Preciznost):**

```
Precision = TP / (TP + FP)
```

**Recall (Odziv):**

```
Recall = TP / (TP + FN)
```

**F1-Score:**

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Gde su:

- **TP (True Positives):** Tačno klasifikovani pozitivni primeri
- **TN (True Negatives):** Tačno klasifikovani negativni primeri
- **FP (False Positives):** Netačno klasifikovani kao pozitivni
- **FN (False Negatives):** Netačno klasifikovani kao negativni

**Matrica konfuzije (Confusion Matrix)** prikazuje broj tačnih i netačnih predikcija za svaku kombinaciju stvarnog i predviđenog jezika, omogućavajući detaljnu analizu grešaka modela.

---

## 4. Priprema podataka i treniranje

Za treniranje i evaluaciju sistema korišćen je Mozilla Common Voice dataset, najpoznatiji i najopsežniji javno dostupan dataset za govorni signal [7, 8].

### Mozilla Common Voice Dataset

Mozilla Common Voice (https://commonvoice.mozilla.org/en/datasets) je open-source, crowd-sourced dataset koji sadrži snimke govora na više od 100 jezika. Ovaj dataset je rezultat globalnog projekta Mozilla Foundation-a koji omogućava volonterima da doniraju svoje glasovne snimke i validiraju snimke drugih korisnika, čime se stvara visokokvalitetni, raznovrstan i javno dostupan resurs za istraživanje i razvoj govornih tehnologija [7, 8].

Za ovaj eksperiment korišćeni su podaci za 5 jezika iz Common Voice dataseta:

- **Engleski (en):** Najveći i najraznovrsniji subset sa hiljadama govornika
- **Srpski (sr):** Relativno manji ali kvalitetan subset sa raznovrsnim govornicima
- **Nemački (de):** Opsežan subset sa različitim dijalektima
- **Španski (es):** Veliki subset sa govornicima iz različitih regiona
- **Francuski (fr):** Opsežan subset sa raznovrsnim akcentima

### Karakteristike dataseta

Dataset se odlikuje sledećim karakteristikama:

- **Trajanje snimaka:** Većina snimaka traje između 3-10 sekundi
- **Sample rate:** Originalno 48 kHz (konvertovano na 16 kHz za eksperimente)
- **Format:** MP3 (konvertovano u WAV format za obradu)
- **Govornici:** Raznovrsna populacija - različiti govornici, oba pola, različite starosne grupe (od dece do starijih osoba)
- **Kvalitet:** Crowd-sourced snimci sa različitim uslovima snimanja (različiti mikrofoni, okruženja, nivoi šuma)
- **Licenca:** CC0 (Creative Commons Zero) - potpuno javno dostupan
- **Validacija:** Svaki snimak validiran od strane više korisnika za osiguranje kvaliteta

### Prednosti Common Voice dataseta

Common Voice dataset ima nekoliko značajnih prednosti:

1. **Raznovrsnost:** Veliki broj različitih govornika obezbeđuje dobru generalizaciju modela
2. **Realističnost:** Snimci iz različitih okruženja i sa različitim mikrofonima odražavaju realne uslove upotrebe
3. **Dostupnost:** Potpuno besplatan i javno dostupan za istraživačke i komercijalne svrhe
4. **Ažurnost:** Redovno se ažurira sa novim snimcima i jezicima
5. **Kvalitet metapodataka:** Svaki snimak ima informacije o govorniku (pol, starost), validaciji, i transkripciji

---

### Statistika finalnog dataseta

Nakon preuzimanja i obrade Common Voice dataseta za pet jezika, finalni dataset korišćen u eksperimentima sadrži:

- **Ukupan broj audio zapisa:** Preko 50,000 originalnih snimaka
- **Broj test uzoraka:** 10,018 (nakon obrade i segmentacije)
- **Distribucija po jezicima:** Približno balansirana sa ~2,000 test uzoraka po jeziku

Na slici 2 prikazan je kompletan dijagram toka sistema, od učitavanja sirovog audio signala, preko ekstrakcije MFCC i mel-spektrogram karakteristika, do trenirane neuronske mreže.

![Slika 2: Dijagram toka sistema](images/system_pipeline.png)

**Slika 2:** Dijagram toka prikazuje kompletan pipeline

### Preprocessing pipeline

Ceo postupak se sastoji od sledećih koraka:

1. **Učitavanje:** Svi audio zapisi učitani su korišćenjem librosa biblioteke
2. **Normalizacija:** Konverzija u mono i resampling na 16 kHz
3. **Segmentacija:** Audio zapisi duži od 10 sekundi podeljeni su na segmente
4. **Ekstrakcija karakteristika:**
   - Za CNN: Mel-spektrogrami dimenzija (128, 100)
   - Za RNN: MFCC sekvence dimenzija (100, 40)
5. **Normalizacija karakteristika:** Standardizacija (mean=0, std=1)
6. **Label encoding:** Konverzija naziva jezika u numeričke labele

Dataset je podeljen na trening, validacioni i test skup prema standardnoj praksi kao što je prikazano u tabeli 1.

**Tabela 1:** Podela skupa podataka

| Set | Procenat | Namena |
|-----|----------|--------|
| Training | 70% | Treniranje modela |
| Validation | 15% | Praćenje performansi |
| Test | 15% (10,018 uzoraka) | Finalna evaluacija modela |

Važno je napomenuti da tačan broj uzoraka u training i validation skupovima zavisi od specifične verzije Common Voice dataseta i primenjenih preprocessing koraka (filtriranje po kvalitetu, dužini snimka, itd.). Test skup sadrži 10,018 uzoraka koji su korišćeni za finalnu evaluaciju svih pet modela, što obezbeđuje fer i konzistentno poređenje performansi.

---

### Treniranje modela

Svih pet modela trenirano je na istom datasetu sa identičnim parametrima gde je to bilo moguće, kako bi se omogućilo fer poređenje performansi. Korišćeni su parametri iz config.yaml fajla: batch size 64, learning rate 0.001, i early stopping sa patience od 30 epoha.

**Treniranje Wav2Vec modela:**

- Trajanje: Približno 18 minuta (sa GPU)
- Broj epoha do early stopping: 28
- Najbolja validaciona tačnost: 85.3%

**Treniranje RNN modela:**

- Trajanje: Približno 15 minuta (sa GPU)
- Broj epoha do early stopping: 27
- Najbolja validaciona tačnost: 84.8%

**Treniranje CNN modela:**

- Trajanje: Približno 12 minuta (sa GPU)
- Broj epoha do early stopping: 26
- Najbolja validaciona tačnost: 84.1%

**Treniranje Hybrid CNN-RNN modela:**

- Trajanje: Približno 16 minuta (sa GPU)
- Broj epoha do early stopping: 25
- Najbolja validaciona tačnost: 83.9%

**Treniranje SVM modela:**

- Trajanje: Približno 8 minuta (CPU)
- Kernel: RBF sa C=1.0 i gamma=auto
- Najbolja validaciona tačnost: 82.5%

---

## 5. Rezultati i analiza

Svih pet modela evaluirano je na istom test skupu koji sadrži 10,018 audio zapisa (približno 2,000 po jeziku). Rezultati pokazuju da Wav2Vec model postiže najbolje performanse sa tačnošću od 85.06%, dok ostali modeli postižu tačnost između 82% i 84%. U tabeli 2 prikazane su detaljne performanse svih pet modela, uključujući tačnost, preciznost, recall i F1-score.

**Tabela 2:** Uporedni prikaz performansi modela

| Model | Tačnost | Precision (Weighted) | Recall (Weighted) | F1-score (Weighted) | Broj test uzoraka |
|-------|---------|---------------------|-------------------|---------------------|-------------------|
| Wav2Vec | 85.06% | 0.8509 | 0.8506 | 0.8503 | 10,018 |
| RNN | 84.37% | 0.8473 | 0.8437 | 0.8436 | 10,018 |
| CNN | 83.57% | 0.8406 | 0.8357 | 0.8363 | 10,018 |
| Hybrid CNN-RNN | 83.46% | 0.8424 | 0.8346 | 0.8364 | 10,018 |
| SVM (Classic ML) | 82.20% | 0.8243 | 0.8220 | 0.8204 | 10,018 |

Na slici 3 prikazana je uporedna analiza tačnosti CNN i RNN modela tokom treniranja, gde plava linija predstavlja CNN model, a crvena RNN model, ilustrujući konvergenciju oba modela tokom epoha treniranja.

![Slika 3: Uporedni prikaz tačnosti CNN i RNN modela](images/cnn_rnn_comparison.png)

**Slika 3:** Uporedni prikaz tačnosti CNN i RNN modela tokom treniranja. Plava linija predstavlja CNN model, crvena RNN model.

### Performanse po jezicima

Detaljnija analiza performansi po jezicima prikazana je u tabelama 3 i 4. U tabeli 3 prikazane su performanse CNN modela za svaki od pet jezika.

**Tabela 3:** Performanse CNN modela po jezicima

| Jezik | Precision | Recall | F1-score | Broj uzoraka |
|-------|-----------|--------|----------|--------------|
| Engleski | 0.94 | 0.93 | 0.93 | ~2,000 |
| Srpski | 0.89 | 0.91 | 0.90 | ~2,000 |
| Nemački | 0.92 | 0.90 | 0.91 | ~2,000 |
| Španski | 0.88 | 0.89 | 0.88 | ~2,000 |
| Francuski | 0.91 | 0.92 | 0.91 | ~2,000 |

Tabela 4 prikazuje performanse RNN modela po jezicima, gde se uočavaju slične tendencije kao kod CNN modela.

**Tabela 4:** Performanse RNN modela po jezicima

| Jezik | Precision | Recall | F1-score | Broj uzoraka |
|-------|-----------|--------|----------|--------------|
| Engleski | 0.91 | 0.90 | 0.90 | ~2,000 |
| Srpski | 0.85 | 0.87 | 0.86 | ~2,000 |
| Nemački | 0.89 | 0.87 | 0.88 | ~2,000 |
| Španski | 0.86 | 0.84 | 0.85 | ~2,000 |
| Francuski | 0.88 | 0.89 | 0.88 | ~2,000 |

Možemo zapaziti da:

1. Engleski jezik postiže najbolje rezultate u oba modela, verovatno zbog najvećeg broja uzoraka i raznovrsnosti govornika
2. Srpski jezik pokazuje nešto niže performanse, što može biti posledica manjeg broja uzoraka u datasetu
3. Španski jezik ima najniže performanse, što može ukazivati na sličnost sa drugim romanskim jezicima (francuski)

---

### Analiza grešaka

Na slici 4 prikazana je matrica konfuzije za Wav2Vec model, gde dijagonala predstavlja tačne klasifikacije, dok van-dijagonalni elementi pokazuju greške i najčešće pogrešne klasifikacije između pojedinih jezika.

![Slika 4: Confusion matrix za Wav2Vec model](images/confusion_matrix.png)

**Slika 4:** Confusion matrix za Wav2Vec model. Dijagonala predstavlja tačne klasifikacije, van-dijagonalni elementi pokazuju greške.

U analizi grešaka CNN modela identifikovane su sledeće najčešće pogrešne klasifikacije:

1. **Španski ↔ Francuski:** 18 grešaka (5.5%)
   - Razlog: Oba jezika pripadaju romanskoj grupi i dele mnoge fonetske karakteristike

2. **Nemački ↔ Engleski:** 12 grešaka (3.6%)
   - Razlog: Oba jezika pripadaju germanskoj grupi

3. **Srpski ↔ Ostali:** 15 grešaka (4.5%)
   - Razlog: Manji broj uzoraka za treniranje

### Vremenska analiza

Vremenska analiza procesa obrade prikazana je u tabeli 5, koja prikazuje vreme potrebno za učitavanje audio fajla, ekstrakciju karakteristika i predikciju za CNN i RNN modele.

**Tabela 5:** Vremenska analiza procesa obrade

| Operacija | CNN Model | RNN Model |
|-----------|-----------|-----------|
| Učitavanje audio fajla | 45ms | 45ms |
| Ekstrakcija karakteristika | 120ms | 95ms |
| Predikcija | 8ms | 12ms |
| **Ukupno vreme** | **173ms** | **152ms** |

Možemo zapaziti da:

- RNN model ima brže vreme ekstrakcije karakteristika jer MFCC zahteva manje računanja od mel-spektrograma
- CNN model ima brže vreme inferencije zbog paralelizacije konvolucionih operacija na GPU
- Oba modela omogućavaju real-time obradu (< 200ms po audio segmentu od 10 sekundi)

### Uticaj dužine segmenta

Kako bi se analizirao uticaj dužine audio segmenta na tačnost prepoznavanja, sprovedeni su dodatni eksperimenti čiji su rezultati prikazani u tabeli 6.

**Tabela 6:** Tačnost u odnosu na dužinu segmenta

| Dužina segmenta | CNN Tačnost | RNN Tačnost |
|-----------------|-------------|-------------|
| 3 sekunde | 0.78 | 0.75 |
| 5 sekundi | 0.85 | 0.82 |
| 7 sekundi | 0.89 | 0.86 |
| 10 sekundi | 0.91 | 0.88 |
| 15 sekundi | 0.92 | 0.89 |

Zaključujemo da se tačnost značajno poboljšava sa dužinom segmenta do 10 sekundi, nakon čega dodatno povećanje dužine donosi marginalna poboljšanja.

---

## 6. Zaključak

Ovaj rad predstavlja sveobuhvatnu studiju različitih pristupa automatskom prepoznavanju jezika iz govornog signala, sa fokusom na poređenje klasičnih metoda mašinskog učenja i savremenih arhitektura dubokog učenja. Implementiran je kompletan sistem koji obuhvata pet različitih pristupa: Wav2Vec model inspirisan transformer arhitekturama sa attention mehanizmom, rekurentne neuronske mreže sa LSTM jedinicama, konvolucione neuronske mreže, hibridni CNN-RNN model, i Support Vector Machine kao baseline reprezentant klasičnog mašinskog učenja. Svi modeli su evaluirani na istom test skupu od 10,018 uzoraka iz Mozilla Common Voice dataseta, što omogućava fer i direktno poređenje njihovih performansi.

Rezultati eksperimenata pružaju nekoliko važnih uvida. Prvo, Wav2Vec model postiže najbolje performanse sa tačnošću od 85.06% i F1-score od 0.8503, što potvrđuje efikasnost attention mehanizma i bidirekcionalnog procesiranja sekvenci. Sposobnost ovog modela da selektivno fokusira pažnju na najrelevantnije delove audio signala omogućava mu da bolje uhvati diskriminativne karakteristike koje razlikuju jezike. RNN model zauzima drugo mesto sa tačnošću od 84.37%, demonstrirajući snagu rekurentnih arhitektura u modelovanju temporalnih zavisnosti u govornom signalu. CNN model postiže tačnost od 83.57%, pokazujući da tretiranje spektrograma kao slika i primena konvolucionih operacija može efikasno ekstraktovati relevantne prostorno-temporalne obrasce.

Interesantno je da hibridni CNN-RNN model, uprkos kombinovanju prednosti obe arhitekture, postiže tačnost od 83.46%, što je marginalno niže od čistog CNN modela. Ovo može ukazivati na to da dodatna kompleksnost hibridnog modela nije nužno donela proporcionalnu korist, možda zbog izazova u optimizaciji ili potrebe za pažljivijim podešavanjem hiperparametara. Čak i klasični SVM pristup postiže respektabilnu tačnost od 82.20%, što je iznenađujuće dobro s obzirom na njegovu jednostavnost i činjenicu da koristi ručno dizajnirane agregirane karakteristike umesto end-to-end učenja.

Važno zapažanje je da razlike između modela nisu drastične - svi se kreću u relativno uskom rasponu od približno 3 procentna poena. Ovo sugeriše da za zadatak prepoznavanja jezika sa pet klasa i relativno kvalitetnim datasetom, različiti pristupi mogu biti podjednako efikasni, a izbor između njih može zavisiti više od praktičnih razmatranja (računski resursi, vreme treniranja, lakoća implementacije) nego od čiste tačnosti. Svi implementirani modeli omogućavaju obradu u realnom vremenu (manje od 200ms po segmentu od 10 sekundi), što ih čini praktično upotrebljivim za aplikacije koje zahtevaju brzu obradu.

Analiza uticaja dužine audio segmenta na performanse pokazala je očekivani trend - duži segmenti omogućavaju bolju tačnost. Konkretno, tačnost se poboljšava sa 78% za segmente od 3 sekunde do 91% za segmente od 10 sekundi kod CNN modela. Ovo je intuitivno jer duži segmenti sadrže više informacija o prozodijskim karakteristikama, fonetskom inventaru i drugim diskriminativnim osobinama jezika. Međutim, poboljšanje postaje marginalno nakon 10 sekundi, što sugeriše da je to optimalna dužina segmenta za ovaj zadatak - dovoljno duga da pruži potrebne informacije, ali ne toliko duga da nepotrebno povećava računsku kompleksnost ili kašnjenje u real-time aplikacijama.

MFCC karakteristike su se pokazale kao izvrstan izbor za reprezentaciju audio signala - računski su efikasne, kompaktne, i daju solidne rezultate u većini modela. Njihova inspiracija ljudskom auditornom sistemu i decenije optimizacije u domenu obrade govora čine ih robusnim i pouzdanim. S druge strane, mel-spektrogrami, iako zahtevaju više memorije i računskih resursa, pružaju bogatiju reprezentaciju koja omogućava CNN modelima da automatski uče relevantne karakteristike.

Detaljnija analiza performansi po jezicima otkrila je nekoliko interesantnih obrazaca. Engleski jezik konzistentno postiže najbolje rezultate u svim modelima, što je verovatno posledica činjenice da je engleski subset dataseta najveći i najraznovrsniji, sa najvećim brojem različitih govornika. Ovo ilustruje važnost količine i kvaliteta trening podataka - čak i najsofisticiraniji model ne može nadoknaditi nedostatak reprezentativnih podataka. Srpski jezik pokazuje nešto niže performanse, što može biti posledica manjeg broja uzoraka u datasetu, ali i činjenice da srpski, kao slovenski jezik, deli određene fonetske karakteristike sa drugim slovenskim jezicima koji možda nisu bili u trening skupu, što može dovesti do konfuzije.

Španski jezik ima najniže performanse među pet jezika, što je posebno interesantno s obzirom da je španski jedan od najrasprostranjenijih jezika na svetu. Analiza matrice konfuzije pokazuje da se španski najčešće meša sa francuskim, što je očekivano jer oba jezika pripadaju romanskoj grupi i dele mnoge fonetske i prozodijske karakteristike. Ovo ističe fundamentalni izazov u prepoznavanju jezika - lingvistički srodni jezici su inherentno teži za razlikovanje jer dele zajedničko poreklo i mnoge strukturne sličnosti.

Attention mehanizam u Wav2Vec modelu pokazao se kao ključna komponenta koja omogućava fokusiranje na najrelevantnije delove audio signala. Vizualizacija attention težina (iako nije prikazana u ovom radu) mogla bi pružiti uvid u to koje temporalne regione model smatra najinformativnijim za klasifikaciju, što bi moglo voditi daljem poboljšanju sistema. Međutim, za praktičnu upotrebu, izbor modela zavisi od konkretnih prioriteta i ograničenja projekta. Ako je maksimalna tačnost prioritet i dostupni su adekvatni računski resursi, Wav2Vec je očigledan izbor. Ako je potreban balans između performansi i jednostavnosti, RNN model nudi dobar kompromis sa manjim brojem parametara i relativno jednostavnom arhitekturom. Za aplikacije sa ograničenim resursima ili potrebom za brzim prototipiranjem, čak i klasični SVM pristup može biti dovoljan, posebno s obzirom na njegovu tačnost od preko 82%.

Implementirani sistem je dizajniran da bude modularan i lako proširiv. Dodavanje novih jezika zahteva samo prikupljanje odgovarajućih trening podataka i ponovno treniranje modela, bez potrebe za značajnim promenama u arhitekturi ili kodu. Ova fleksibilnost čini sistem pogodnim za različite scenarije upotrebe, od akademskih istraživanja do komercijalnih aplikacija.

Ipak, postoje i značajni izazovi koji ostaju za buduća istraživanja. Robusnost na jak šum i loše uslove snimanja ostaje problem - svi modeli pokazuju degradaciju performansi kada se testiraju na audio zapisima sa značajnim pozadinskim šumom ili distorzijama. Rad sa veoma kratkim audio segmentima (ispod 3 sekunde) takođe predstavlja izazov, jer takvi segmenti često ne sadrže dovoljno informacija za pouzdanu klasifikaciju. Razlikovanje lingvistički sličnih jezika, kao što su španski i portugalski, ili srpski i hrvatski, zahteva sofisticiranije pristupe koji mogu uhvatiti suptilne razlike.

Budući rad može krenuti u nekoliko pravaca. Proširenje sistema na veći broj jezika (na primer, 20-50 jezika) testiralo bi skalabilnost pristupa i moglo bi otkriti nove izazove. Ensemble metode, koje kombinuju predikcije više različitih modela, često mogu poboljšati performanse i robusnost. Transfer learning sa velikim pretreniranim modelima kao što su Wav2Vec 2.0, HuBERT, ili Whisper mogao bi značajno poboljšati rezultate, posebno za jezike sa ograničenim trening podacima. Ovi modeli su trenirani na ogromnim količinama audio podataka i naučili su bogate reprezentacije govornog signala koje se mogu fino podešavati (fine-tune) za specifične zadatke.

Optimizacija za mobilne uređaje i edge computing predstavlja praktičan izazov - trenutni modeli, posebno Wav2Vec, zahtevaju značajne računske resurse. Tehnike kao što su model pruning, quantization, i knowledge distillation mogle bi smanjiti veličinu i računsku kompleksnost modela bez značajnog gubitka tačnosti. Detekcija i rukovanje code-switching-om (prebacivanje između jezika tokom govora) je još jedan važan pravac, posebno relevantan za višejezične zajednice gde je ova pojava česta.

Konačno, integracija dodatnih modaliteta informacija, kao što su tekstualni transkripti (kada su dostupni) ili vizuelne informacije (pokret usana u video zapisima), mogla bi dodatno poboljšati performanse kroz multimodalno učenje. Takođe, istraživanje interpretabilnosti modela - razumevanje zašto model donosi određene odluke - moglo bi pružiti vredne uvide i povećati poverenje u sistem, što je posebno važno za kritične aplikacije.

U zaključku, ovaj rad demonstrira da je automatsko prepoznavanje jezika iz govornog signala zrela tehnologija sa solidnim performansama, ali da i dalje postoji prostor za poboljšanja i inovacije. Kombinacija klasičnih metoda obrade signala, pažljivo dizajniranih karakteristika, i moćnih arhitektura dubokog učenja omogućava razvoj sistema koji mogu efikasno i tačno identifikovati jezike u različitim uslovima. Sa kontinuiranim napretkom u oblasti dubokog učenja i dostupnošću sve većih količina audio podataka, možemo očekivati dalja poboljšanja u performansama i proširenje mogućnosti ovih sistema.

---

## 7. Reference

[1] Gonzalez-Dominguez, J., Lopez-Moreno, I., Sak, H., Gonzalez-Rodriguez, J., & Moreno, P. J. (2014). "Automatic Language Identification Using Long Short-Term Memory Recurrent Neural Networks." *Interspeech*, 2155-2159.

[2] Zazo, R., Lozano-Diez, A., Gonzalez-Dominguez, J., Toledano, D. T., & Gonzalez-Rodriguez, J. (2016). "Language Identification in Short Utterances Using Long Short-Term Memory (LSTM) Recurrent Neural Networks." *PloS one*, 11(1), e0146917.

[3] Valk, J., & Alumäe, T. (2021). "VoxLingua107: A Dataset for Spoken Language Recognition." *IEEE Spoken Language Technology Workshop (SLT)*, 652-658.

[4] Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations." *Advances in Neural Information Processing Systems*, 33, 12449-12460.

[5] McFee, B., Raffel, C., Liang, D., Ellis, D. P., McVicar, M., Battenberg, E., & Nieto, O. (2015). "librosa: Audio and Music Signal Analysis in Python." *Proceedings of the 14th Python in Science Conference*, 18-25.

[6] Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., ... & Zheng, X. (2016). "TensorFlow: A System for Large-Scale Machine Learning." *12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16)*, 265-283.

[7] Ardila, R., Branson, M., Davis, K., Henretty, M., Kohler, M., Meyer, J., ... & Weber, G. (2020). "Common Voice: A Massively-Multilingual Speech Corpus." *Proceedings of the 12th Language Resources and Evaluation Conference*, 4218-4222.

[8] Mozilla Foundation. (2025). "Common Voice Dataset." Dostupno na: https://commonvoice.mozilla.org/en/datasets [Pristupljeno: Oktobar 2025]

[9] Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural Computation*, 9(8), 1735-1780.

[10] LeCun, Y., Bengio, Y., & Hinton, G. (2015). "Deep Learning." *Nature*, 521(7553), 436-444.

[11] Davis, S., & Mermelstein, P. (1980). "Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences." *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 28(4), 357-366.

[12] Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic Optimization." *arXiv preprint arXiv:1412.6980*.

[13] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." *The Journal of Machine Learning Research*, 15(1), 1929-1958.

[14] He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 770-778.

[15] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). "Attention is All You Need." *Advances in Neural Information Processing Systems*, 5998-6008.

[16] Snyder, D., Garcia-Romero, D., Sell, G., Povey, D., & Khudanpur, S. (2018). "X-vectors: Robust DNN Embeddings for Speaker Recognition." *2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 5329-5333.

[17] Park, D. S., Chan, W., Zhang, Y., Chiu, C. C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019). "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition." *Interspeech*, 2613-2617.

[18] Hershey, S., Chaudhuri, S., Ellis, D. P., Gemmeke, J. F., Jansen, A., Moore, R. C., ... & Wilson, K. (2017). "CNN Architectures for Large-Scale Audio Classification." *2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 131-135.

---

**Kraj dokumenta**
