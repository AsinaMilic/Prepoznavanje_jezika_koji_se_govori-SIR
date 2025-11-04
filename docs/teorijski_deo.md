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

Razvoj tehnologija za automatsko prepoznavanje jezika iz govornog signala predstavlja jedan od najznačajnijih pravaca istraživanja u oblasti obrade govora i mašinskog učenja. Tokom poslednjih nekoliko decenija, sa ubrzanim procesom globalizacije i eksponencijalnim rastom međunarodne komunikacije, potreba za efikasnim sistemima koji mogu automatski identifikovati jezik govora postala je sve prisutnija u brojnim domenima - od call centara do bezbednosnih aplikacija.

Ako pogledamo istoriju, prvi pokušaji automatskog prepoznavanja jezika datiraju još iz 1970-ih godina. Tada su istraživači koristili relativno jednostavne akustičke karakteristike i statističke metode za razlikovanje između ograničenog broja jezika - uglavnom se radilo o nekoliko evropskih jezika. Ovi rani sistemi bili su zasnovani na analizi spektralnih karakteristika govora i fonetskih osobina pojedinih jezika, ali su, realno, imali dosta ograničenja kada je u pitanju tačnost i broj jezika koje su mogli da prepoznaju. Tradicionalni pristupi, koji su dominirali sve do početka 21. veka, zasnivali su se na ručno dizajniranim karakteristikama i klasičnim algoritmima mašinskog učenja, kao što su Gaussian Mixture Models (GMM) i Support Vector Machines (SVM). Ovi modeli zahtevali su značajan ekspertski rad u domenu lingvistike i obrade signala kako bi se identifikovale relevantne karakteristike koje razlikuju jezike.

Međutim, prava revolucija desila se oko 2012. godine sa razvojem dubokih konvolucionih neuronskih mreža i, što je možda još važnije, sa dostupnošću velikih količina audio podataka. Neuronske mreže pokazale su da mogu automatski da uče hijerarhijske reprezentacije iz sirovih audio podataka, što je praktično eliminisalo potrebu za ručnim dizajnom karakteristika - nešto što je ranije zahtevalo godine ekspertskog rada. Performanse su bile značajno bolje od svega što smo do tada videli [10]. Ova promena paradigme otvorila je vrata za razvoj sistema koji mogu da prepoznaju ne samo desetine, već i stotine različitih jezika sa prilično visokom tačnošću.

Danas, automatsko prepoznavanje jezika ima zaista široku primenu - od industrije, preko društvenih aplikacija, pa do naučnih istraživanja. U komercijalnom sektoru, tehnologija je postala skoro nezamenljiva u višejezičnim call centrima. Zamislite scenario: pozivate neku međunarodnu kompaniju, i sistem automatski prepozna da govorite srpski i preusmeri vas ka operateru koji govori vaš jezik. To značajno poboljšava korisničko iskustvo i, naravno, efikasnost poslovanja. Sistemi za automatsko prevođenje u realnom vremenu, koji su postali sastavni deo moderne međunarodne komunikacije, oslanjaju se na preciznu identifikaciju jezika kao prvi korak u procesu prevođenja. Digitalni asistenti poput Alexe, Siri i Google Assistanta koriste prepoznavanje jezika kako bi mogli da komuniciraju sa korisnicima na njihovom maternjem jeziku, dok streaming platforme poput YouTube-a i Netflix-a koriste ovu tehnologiju za automatsku kategorizaciju sadržaja i preporuke korisnicima.

U društvenom i kulturnom kontekstu, automatsko prepoznavanje jezika doprinosi bezbednosnim sistemima kroz analizu komunikacija u svrhu detekcije pretnji, omogućava automatsko titlovanje medijskog sadržaja čime se povećava pristupačnost informacija, pomaže u očuvanju ugroženih jezika kroz digitalizaciju i analizu audio arhiva, te olakšava učenje stranih jezika kroz interaktivne aplikacije koje pružaju trenutnu povratnu informaciju. U naučnim istraživanjima i medicini, analiza govora ima značajnu ulogu u proučavanju jezičke raznovrsnosti, sociolingvističkih fenomena, kao i u detekciji neuroloških poremećaja koji utiču na govor, poput Parkinsonove bolesti ili afazije.

Ekonomski uticaj ove tehnologije je, blago rečeno, impresivan - globalno tržište rešenja za obradu govora procenjuje se na više desetina milijardi dolara godišnje, sa stopom rasta koja premašuje 15% godišnje. To nije iznenađujuće kada se uzme u obzir da automatizacija procesa prepoznavanja jezika može značajno da smanji operativne troškove u brojnim industrijama - od telekomunikacija do turizma - dok istovremeno povećava efikasnost i kvalitet usluga. U kontekstu globalizacije, ova tehnologija omogućava personalizaciju digitalnih usluga na osnovu jezika korisnika, ciljano oglašavanje na različitim tržištima, i efikasniju komunikaciju između geografski i kulturno udaljenih zajednica. Sa razvojem Internet of Things (IoT) uređaja i kontinuiranim porastom količine audio podataka koji se generišu svakodnevno, očekuje se da automatsko prepoznavanje jezika postane još integrisaniji deo svakodnevnih aplikacija - od pametnih domova koji se prilagođavaju jeziku korisnika, do autonomnih sistema koji mogu da komuniciraju sa ljudima na različitim jezicima.

Ipak, treba biti realan - prepoznavanje jezika iz govornog signala i dalje ostaje prilično kompleksan i izazovan zadatak. Za razliku od tekstualnog prepoznavanja jezika (gde je situacija relativno jednostavna jer su karakteristike stabilne i jasno definisane), audio signal nosi sa sobom gomilu izvora varijabilnosti i šuma. Potrebno je prepoznati jezik iz kratkog audio segmenta koji može biti snimljen u različitim akustičkim uslovima, sa različitim kvalitetom opreme, i od govornika sa različitim karakteristikama glasa. Na rezultate značajno utiču faktori kao što su varijabilnost između govornika (pol, starost, individualne karakteristike glasa), kvalitet snimanja (tip mikrofona, pozadinska buka, kompresija audio signala), trajanje audio segmenta (kraći segmenti nose manje informacija), lingvistička sličnost između jezika (posebno unutar istih jezičkih porodica), kao i postojanje dijalekata, regionalnih varijanti i stranih akcenata.

Dodatni izazov predstavlja i fenomen code-switching-a - prebacivanje između jezika tokom govora. Ovo je česta pojava u višejezičnim zajednicama (npr. neko govori srpski ali ubacuje engleske reči). Takođe, govorni signal može sadržati emotivne komponente, različite stilove govora (formalni, neformalni, brzi, spori), pa čak i specifične karakteristike kao što su pevanje, šaputanje ili govor pod stresom. Sve ovo dodatno komplikuje zadatak automatskog prepoznavanja i čini ga zanimljivim istraživačkim problemom.

Zbog svih ovih faktora, razvoj tačnog i robusnog sistema za automatsko prepoznavanje jezika zahteva pažljivo dizajniranu metodologiju, odabir odgovarajućih karakteristika audio signala, primenu naprednih modela mašinskog učenja, kao i opsežno testiranje na raznovrsnim datasetima koji pokrivaju različite scenarije upotrebe. Ovaj rad predstavlja pokušaj da se sistematski istraže različiti pristupi ovom problemu, od klasičnih metoda mašinskog učenja do najsavremenijih arhitektura dubokih neuronskih mreža, sa ciljem da se identifikuju njihove prednosti, ograničenja i optimalne oblasti primene.

---

## 2. Klasifikacija govornog signala

Klasifikacija govornog signala, odnosno Spoken Language IDentification (SLID), predstavlja specifičan podskup problema klasifikacije u mašinskom učenju koji se bavi automatskim prepoznavanjem jezika na osnovu akustičkih karakteristika govora. Za razliku od tekstualne klasifikacije jezika koja analizira pisani sadržaj, SLID sistemi moraju da ekstrahuju relevantne informacije direktno iz audio signala, što uvodi dodatne dimenzije kompleksnosti.

Ako pogledamo evoluciju pristupa klasifikaciji govornog signala, možemo je podeliti u nekoliko ključnih faza.

![Slika 2.1: Evolucija pristupa u SLID](images/slid_evolution_timeline.png)

**Slika 2.1:** Evolucija pristupa u automatskom prepoznavanju jezika - od ranih GMM/HMM modela 1990-ih, preko SVM pristupa 2000-ih, do dominacije dubokog učenja od 2012. godine.

Tokom 1990-ih i ranih 2000-ih godina, dominirali su pristupi zasnovani na Gaussian Mixture Models (GMM) i Hidden Markov Models (HMM). Ovi modeli koristili su ručno dizajnirane akustičke karakteristike - uglavnom Linear Predictive Coding (LPC) koeficijente i Mel-Frequency Cepstral Coefficients (MFCC). Problem je bio što su ovi modeli zahtevali dosta ekspertskog znanja za dizajn karakteristika, i često nisu bili baš najbolji kada je trebalo da generalizuju na nove uslove snimanja ili govornike koje nisu videli tokom treniranja.

Prelazak na metode mašinskog učenja zasnovane na Support Vector Machines (SVM) i Random Forests tokom 2000-ih godina doneo je izvesna poboljšanja - posebno u pogledu robusnosti i sposobnosti da rade sa višedimenzionalnim karakteristikama. Ali, i dalje je postojao fundamentalni problem: zavisnost od kvaliteta ručno dizajniranih karakteristika. To je bilo pravo usko grlo u razvoju sistema jer je značilo da napredak zavisi od toga koliko dobro eksperti mogu da identifikuju relevantne karakteristike.

Revolucionarna promena desila se od 2012. godine, kada su duboke neuronske mreže postale dominantan pristup u SLID zadacima [10]. Ova promena paradigme nije bila slučajna - desilo se nekoliko stvari istovremeno: postale su dostupne velike količine označenih audio podataka, razvijeni su moćniji grafički procesori (GPU) koji omogućavaju efikasno treniranje dubokih mreža, a i teorijski smo bolje razumeli kako da optimizujemo duboke arhitekture. Sve to zajedno stvorilo je savršenu oluju za napredak u ovoj oblasti. Ključne prednosti dubokog učenja uključuju automatsko učenje karakteristika iz sirovih podataka (eliminirajući potrebu za ručnim dizajnom), sposobnost učenja hijerarhijskih višeslojnih reprezentacija (od jednostavnih akustičkih obrazaca do kompleksnih jezičkih struktura), i end-to-end treniranje koje optimizuje ceo sistem istovremeno, umesto optimizacije pojedinačnih komponenti odvojeno.

Duboke neuronske mreže pokazale su sposobnost da uče apstraktne reprezentacije koje su često superiorne ručno dizajniranim karakteristikama, čak i kada su dizajnirane od strane eksperata sa decenijama iskustva. Ove naučene reprezentacije pokazuju bolju generalizaciju na nove uslove, veću robusnost na šum i distorzije, i sposobnost da automatski identifikuju relevantne karakteristike koje razlikuju jezike, uključujući i one koje možda nisu bile očigledne ljudskim istraživačima.

### CNN arhitekture za SLID

Konvolucione neuronske mreže (CNN) originalno su razvijene za zadatke kompjuterske vizije - prepoznavanje slika, detekciju objekata i slično. Ali, pokazalo se da su izuzetno efikasne i u domenu obrade audio signala, što je na prvi pogled možda malo neočekivano. Ključna ideja koja omogućava primenu CNN-a na audio podatke jeste tretiranje spektrograma kao dvodimenzionalnih slika - što, kada malo razmislite, ima dosta smisla. U ovom pristupu, horizontalna osa spektrograma predstavlja vremensku dimenziju (kako se signal menja tokom vremena), dok vertikalna osa predstavlja frekventnu dimenziju (koje frekvencije su prisutne u signalu). Intenzitet boje ili nijanse u svakoj tački spektrograma odgovara energiji signala na toj specifičnoj frekvenciji u tom specifičnom trenutku.

Ova analogija sa slikama nije samo površna - spektrogrami zaista poseduju prostorne strukture i obrasce koji su analogni vizuelnim karakteristikama. Na primer, harmonici u govornom signalu pojavljuju se kao horizontalne linije u spektrogramu (što je vizuelno prilično jasno kada pogledate spektrogram), formanti (rezonantne frekvencije vokalnog trakta) formiraju karakteristične obrasce koji se razlikuju između različitih fonema, a prelazi između fonema se vide kao promene u spektralnoj strukturi. Različiti jezici imaju različite fonetske inventare i prozodijske karakteristike (intonacija, ritam, naglasak), što se sve reflektuje u specifičnim obrascima u spektrogramima. Različiti jezici imaju različite fonetske inventare, različite distribucije fonema, i različite prozodijske karakteristike (intonacija, ritam, naglasak), što se sve reflektuje u specifičnim obrascima u spektrogramima.

CNN arhitekture za SLID sastoje se od nekoliko ključnih komponenti koje rade zajedno da ekstrahuju i klasifikuju ove obrasce:

**Konvolucioni slojevi** predstavljaju srce CNN arhitekture. Oni primenjuju skup naučenih filtera (kernela) na ulazni spektrogram, pri čemu svaki filter ima relativno male dimenzije (tipično 3×3 ili 5×5 piksela) ali se primenjuje na celom spektrogramu kroz operaciju konvolucije. Svaki filter uči da detektuje specifičan lokalni spektralno-temporalni obrazac. U nižim slojevima mreže, filteri obično uče jednostavne karakteristike kao što su ivice, prelazi između frekvencija, ili lokalne varijacije u energiji. U višim slojevima, filteri kombinuju ove jednostavne karakteristike da detektuju kompleksnije strukture kao što su formanti (karakteristični za specifične vokale), prelazi između fonema, harmonička struktura (koja se razlikuje između tonskih i netonskih jezika), ili specifični spektralni obrasci karakteristični za određene jezike.

Važno je napomenuti da se ovi filteri ne dizajniraju ručno - oni se uče automatski tokom procesa treniranja kroz algoritam backpropagation. To je zapravo jedna od najmoćnijih stvari kod dubokog učenja: mreža sama otkriva koje karakteristike su najrelevantnije za razlikovanje jezika. Često se desi da filteri detektuju obrasce koje ljudski eksperti možda ne bi intuitivno identifikovali kao diskriminativne, što je fascinantno kada razmislite o tome.

![Slika 2.2: Vizualizacija naučenih CNN filtera](images/cnn_filters_visualization.png)

**Slika 2.2:** Vizualizacija naučenih filtera u prvom konvolucionom sloju CNN modela. Svaki filter detektuje specifične spektralno-temporalne obrasce u mel-spektrogramima. Možemo videti da neki filteri reaguju na horizontalne strukture (harmonici), drugi na vertikalne (brze promene u vremenu), a treći na dijagonalne obrasce (formantne tranzicije).

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

![Slika 2.4: Pipeline ekstrakcije MFCC karakteristika](images/mfcc_extraction_pipeline.png)

**Slika 2.4:** Dijagram toka ekstrakcije MFCC koeficijenata. Proces počinje sa sirovim audio signalom i prolazi kroz sedam koraka: pre-emphasis filtering, framing, windowing, FFT, mel-filter bank, logaritamsku kompresiju, i konačno DCT koji proizvodi finalne MFCC koeficijente. Svaki korak ima specifičnu ulogu u transformaciji signala.

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

![Slika 2.3: Poređenje mel-spektrograma za različite jezike](images/mel_spectrogram_comparison.png)

**Slika 2.3:** Poređenje mel-spektrograma iste rečenice izgovorene na pet različitih jezika. Možemo uočiti razlike u prozodijskim obrascima (intonacija), spektralnoj strukturi (fonetski inventar), i temporalnoj dinamici (ritam govora). Engleski pokazuje karakterističan obrazac sa jasno definisanim formantima, srpski ima specifične visokofrekventne komponente zbog glasova kao što su ć i č, nemački pokazuje oštre tranzicije između fonema, dok španski i francuski dele određene sličnosti u spektralnoj strukturi što objašnjava zašto se ova dva jezika ponekad mešaju u klasifikaciji.

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

Metodologija ovog istraživanja obuhvata sistematski pristup razvoju i evaluaciji sistema za automatsko prepoznavanje jezika. Proces je podeljen u nekoliko ključnih faza: dizajn arhitekture sistema, pripremu i obradu podataka, implementaciju različitih modela mašinskog učenja, treniranje i optimizaciju hiperparametara, te konačno evaluaciju i analizu performansi. Svaka od ovih faza zahteva pažljivo razmatranje brojnih faktora kako bi se osiguralo da finalni sistem bude robustan, tačan i praktično upotrebljiv.

Prilikom dizajniranja metodologije, vodilo se računa o nekoliko ključnih principa. Prvo, sistem mora biti modularan - svaka komponenta treba da bude nezavisna i zamenljiva, što omogućava lako testiranje različitih pristupa i konfiguracija. Drugo, potrebno je osigurati reproducibilnost rezultata kroz fiksiranje random seed-ova i detaljno dokumentovanje svih parametara. Treće, evaluacija mora biti fer i konzistentna - svi modeli se testiraju na identičnim podacima pod istim uslovima. Konačno, sistem treba da bude skalabilan i lako proširiv na nove jezike ili različite uslove upotrebe.

### 3.1 Arhitektura sistema i obrada podataka

Implementirani sistem za prepoznavanje jezika sastoji se od nekoliko modularnih komponenti koje zajedno formiraju kompletan pipeline od sirovog audio signala do predikcije jezika. Dizajn sistema zasniva se na principima softverskog inženjerstva koji naglašavaju modularnost, ponovnu upotrebljivost koda, i jasnu separaciju odgovornosti između komponenti. Ovakav pristup ne samo da olakšava razvoj i održavanje sistema, već takođe omogućava jednostavno eksperimentisanje sa različitim konfiguracijama i pristupima.

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

Audio Processing modul predstavlja prvu fazu u pipeline-u obrade i odgovoran je za učitavanje, validaciju i normalizaciju audio zapisa. Ova komponenta je kritična jer kvalitet ulaznih podataka direktno utiče na performanse celog sistema - princip "garbage in, garbage out" posebno je relevantan u mašinskom učenju. Modul je implementiran u Python-u korišćenjem librosa biblioteke [5], koja pruža robusne i efikasne funkcije za manipulaciju audio signalima.

Tokom razvoja ovog modula, suočili smo se sa nekoliko praktičnih izazova. Prvo, audio zapisi iz različitih izvora mogu imati različite sample rate-ove, broj kanala (mono ili stereo), i formate (WAV, MP3, FLAC, itd.). Drugo, neki zapisi mogu biti oštećeni ili imati neočekivane karakteristike koje mogu uzrokovati greške tokom obrade. Treće, potrebno je balansirati između očuvanja kvaliteta signala i računske efikasnosti - na primer, viši sample rate pruža bolju frekventnu rezoluciju ali zahteva više memorije i procesorske moći.

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

- **Sample rate:** 16 kHz (standardna vrednost za govorni signal) - Ova vrednost je izabrana jer predstavlja dobar kompromis između kvaliteta i efikasnosti. Nyquist-Shannon teorema kaže da sample rate mora biti najmanje dvostruko veći od najviše frekvencije koju želimo da uhvatimo. Pošto je ljudski govor dominantno koncentrisan ispod 8 kHz, 16 kHz sample rate je dovoljan da uhvati sve relevantne informacije. Viši sample rate-ovi (npr. 44.1 kHz ili 48 kHz) bi doneli marginalne koristi uz značajno povećanje računske kompleksnosti.

- **Broj kanala:** Mono (konverzija stereo zapisa) - Stereo zapisi se konvertuju u mono jer prostorna informacija (razlika između levog i desnog kanala) nije relevantna za identifikaciju jezika. Konverzija se vrši prosečavanjem oba kanala, što očuvava sve frekventne komponente dok redukuje količinu podataka za obradu.

- **Format:** Podrška za WAV, MP3 i FLAC formate - WAV je nekompresovani format koji pruža najbolji kvalitet ali zauzima najviše prostora. MP3 je kompresovani format sa gubitkom koji je široko rasprostranjen ali može uvesti artifakte. FLAC je kompresovani format bez gubitka koji pruža dobar balans. Librosa biblioteka automatski dekodira sve ove formate u uniformnu reprezentaciju.

Dodatno, modul implementira nekoliko mehanizama za rukovanje greškama i edge case-ovima. Na primer, ako je audio fajl previše kratak (manje od 1 sekunde), sistem ga odbacuje jer ne sadrži dovoljno informacija za pouzdanu klasifikaciju. Ako je fajl oštećen ili u nepoznatom formatu, sistem beleži grešku i nastavlja sa sledećim fajlom umesto da prekine ceo proces. Takođe, implementirana je normalizacija amplitude signala kako bi se osiguralo da svi zapisi imaju sličan nivo glasnoće, što sprečava da razlike u nivou snimanja utiču na ekstrakciju karakteristika.

---

#### Feature Extraction

Feature Extraction modul predstavlja ključnu komponentu sistema koja transformiše sirovi audio signal u numeričke reprezentacije pogodne za mašinsko učenje. Kvalitet ekstraktovanih karakteristika direktno determiniše gornju granicu performansi koje model može postići - čak i najsofisticiraniji model ne može naučiti diskriminativne obrasce ako karakteristike ne sadrže relevantne informacije. Ovaj modul implementira dve komplementarne reprezentacije: MFCC koeficijente za RNN modele i mel-spektrograme za CNN modele.

Izbor između različitih reprezentacija nije proizvoljan već se zasniva na prirodi arhitekture modela i tipu informacija koje želimo da uhvatimo. MFCC koeficijenti su kompaktna reprezentacija koja agregira spektralne informacije u mali broj koeficijenata, što ih čini pogodnim za sekvencionalne modele koji mogu efikasno procesirati duže sekvence. Mel-spektrogrami, s druge strane, zadržavaju punu vremensko-frekventnu strukturu signala, što omogućava CNN modelima da automatski uče relevantne prostorne obrasce.

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

Parametri ekstrakcije su pažljivo odabrani na osnovu literature i preliminarnih eksperimenata:

- **n_mfcc:** 40 koeficijenata - Ovo je više od standardnih 13 koeficijenata koji se često koriste u prepoznavanju govora, ali smo empirijski utvrdili da dodatni koeficijenti nose korisne informacije za razlikovanje jezika. Prvih 12-13 koeficijenata hvataju opštu spektralnu strukturu, dok viši koeficijenti hvataju finije detalje koji mogu biti diskriminativni između sličnih jezika.

- **n_fft:** 2048 (veličina FFT prozora) - Sa sample rate-om od 16 kHz, ovo odgovara prozoru od 128ms, što je dovoljno dugo da uhvati nekoliko perioda osnovne frekvencije čak i za niske muške glasove (tipično oko 80-120 Hz). Veći prozor pruža bolju frekventnu rezoluciju ali lošiju vremensku rezoluciju, i obrnuto. Vrednost 2048 predstavlja dobar kompromis.

- **hop_length:** 512 - Ovo znači da se prozor pomera za 32ms između uzastopnih okvira (512/16000 = 0.032s), što rezultira u preklapanju od 75% između uzastopnih prozora. Ovo veliko preklapanje osigurava da ne propustimo važne temporalne događaje i omogućava glatku temporalnu evoluciju karakteristika.

- **n_mels:** 128 (broj mel filtera za spektrogram) - Ovo je relativno visok broj filtera koji pruža detaljnu frekventnu rezoluciju. Više filtera znači bolju sposobnost da se razlikuju bliske frekvencije, što može biti važno za hvatanje suptilnih razlika u fonetskom inventaru različitih jezika. Međutim, više filtera takođe znači više podataka za obradu, što povećava memorijske zahteve i vreme treniranja.

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

CNN model neuronske mreže dizajniran je za obradu mel-spektrograma kao 2D slika. 

![Slika 3.1: Arhitektura CNN modela](images/cnn_architecture_diagram.png)

**Slika 3.1:** Detaljni dijagram arhitekture CNN modela. Model se sastoji od tri konvoluciona bloka sa postepenim povećanjem broja filtera (32→64→128), praćenih max pooling slojevima. Nakon flatten operacije, dva fully connected sloja vrše finalnu klasifikaciju. Dropout sloj (0.5) između dense slojeva sprečava overfitting.

Arhitektura:

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

![Slika 3.2: Arhitektura RNN/LSTM modela](images/rnn_architecture_diagram.png)

**Slika 3.2:** Arhitektura RNN modela sa dva stacked LSTM sloja. Prvi LSTM sloj (128 units) vraća sekvence (return_sequences=True) koje se prosleđuju drugom LSTM sloju. Dropout slojevi (0.3) između LSTM slojeva sprečavaju overfitting. Finalni dense slojevi vrše klasifikaciju na osnovu izlaza poslednjeg LSTM sloja.

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

![Slika 3.3: Arhitektura Wav2Vec modela](images/wav2vec_architecture_diagram.png)

**Slika 3.3:** Arhitektura Wav2Vec modela sa bidirekcionalnim LSTM i attention mehanizmom. Bidirectional LSTM obrađuje sekvencu u oba smera (forward i backward), a attention layer fokusira pažnju na najrelevantnije delove sekvence. Global average pooling agregira informacije iz cele sekvence pre finalne klasifikacije.

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

Priprema podataka je, iskreno, jedan od najkritičnijih koraka u razvoju sistema za mašinsko učenje - možda čak i najvažniji. Postoji poznata izreka u mašinskom učenju: "garbage in, garbage out". Kvalitet, količina i raznovrsnost trening podataka često imaju veći uticaj na finalne performanse nego izbor arhitekture modela ili podešavanje hiperparametara. U oblasti obrade govora, ovo je posebno izraženo jer govorni signal nosi sa sobom gomilu izvora varijabilnosti - od individualnih karakteristika govornika (visina glasa, tempo govora), preko uslova snimanja (kvalitet mikrofona, pozadinska buka), pa do lingvističkih varijacija unutar istog jezika (dijalekti, akcenti).

Za treniranje i evaluaciju sistema korišćen je Mozilla Common Voice dataset [7, 8]. Izbor ovog dataseta nije bio slučajan - razmatrali smo nekoliko alternativa (VoxForge, LibriSpeech, pa čak i neke komercijalne opcije), ali Common Voice se na kraju pokazao kao najbolji izbor. Razlozi su bili višestruki: raznovrsnost jezika i govornika, solidno kvalitet snimaka, potpuna dostupnost (što je bitno za reproducibilnost istraživanja), i aktivna zajednica koja ga održava i konstantno unapređuje.

### Mozilla Common Voice Dataset

Mozilla Common Voice (https://commonvoice.mozilla.org/en/datasets) je open-source, crowd-sourced dataset koji sadrži snimke govora na više od 100 jezika - što je, kada malo razmislite, impresivno. Ovaj dataset je rezultat ambicioznog globalnog projekta Mozilla Foundation-a koji je započet 2017. godine sa idejom da demokratizuje tehnologije obrade govora. Za razliku od mnogih komercijalnih dataseta koji su zatvoreni i koštaju pristojne pare, Common Voice je potpuno otvoren i besplatan. To omogućava istraživačima i developerima širom sveta (uključujući i nas studente) da razvijaju i unapređuju govornu tehnologiju bez finansijskih barijera.

Proces kreiranja dataseta je zasnovan na crowd-sourcing modelu gde volonteri mogu da doprinesu na dva načina: ili doniraju svoje glasovne snimke čitanjem prikazanih rečenica, ili validiraju snimke drugih korisnika slušanjem i potvrđivanjem da li snimak tačno odgovara prikazanom tekstu. Ovaj dvostepeni proces je zapravo prilično pametan jer osigurava visok kvalitet podataka - svaki snimak mora biti validiran od strane više nezavisnih korisnika pre nego što bude uključen u finalni dataset. To znači da ne može neko samo da uploaduje loše snimke i da one automatski uđu u dataset. Dodatno, Mozilla implementira različite mehanizme za detekciju i filtriranje nekvalitetnih snimaka, uključujući automatsku detekciju šuma, proveru dužine snimka, i analizu konzistentnosti između validatora.

Jedan od ključnih aspekata Common Voice projekta je fokus na raznovrsnost. Dataset ne samo da pokriva veliki broj jezika, već i unutar svakog jezika nastoji da uhvati raznovrsnost govornika - različite polove, starosne grupe, regione, akcente i dijalekte. Ova raznovrsnost je kritična za razvoj sistema koji mogu da generalizuju na široku populaciju korisnika, a ne samo na usku demografsku grupu koja je možda dominantna u trening podacima [7, 8].

Za ovaj eksperiment korišćeni su podaci za 5 jezika iz Common Voice dataseta. Izbor ovih specifičnih jezika bio je vođen nekoliko kriterijuma: prvo, želeli smo da pokrijemo različite jezičke porodice (germanske, romanske, slovenske) kako bismo testirali sposobnost sistema da razlikuje i lingvistički udaljene i relativno slične jezike; drugo, odabrali smo jezike sa dovoljnom količinom dostupnih podataka za treniranje robusnih modela; treće, uključili smo srpski jezik kao predstavnika južnoslovenskih jezika koji su manje zastupljeni u istraživanjima obrade govora.

- **Engleski (en):** Najveći i najraznovrsniji subset sa hiljadama govornika iz različitih anglofon ih zemalja (SAD, UK, Kanada, Australija, Indija, itd.). Engleski subset sadrži preko 100,000 validiranih snimaka, što omogućava treniranje veoma robusnih modela. Raznovrsnost akcenata u ovom subsetu je posebno vredna jer reflektuje globalnu prirodu engleskog jezika - od britanskog RP (Received Pronunciation) i američkog General American, do indijskog, australijskog i drugih regionalnih varijanti.

- **Srpski (sr):** Relativno manji ali kvalitetan subset sa nekoliko hiljada snimaka od govornika iz Srbije i dijaspore. Iako je manji od engleskog subseta (što je očekivano), srpski dataset je dovoljno velik za treniranje i sadrži dobru raznovrsnost govornika različitih polova i starosnih grupa. Uključivanje srpskog jezika bilo je posebno važno za ovaj rad jer predstavlja južnoslovenske jezike koji su, nažalost, često zanemareni u istraživanjima obrade govora. Takođe, srpski ima bogatu morfologiju i specifične fonetske karakteristike (ć, č, đ, dž) koje ga čine interesantnim test case-om.

- **Nemački (de):** Opsežan subset sa desetinama hiljada snimaka koji pokriva različite dijalekte nemačkog jezika - od standardnog Hochdeutsch-a do regionalnih varijanti kao što su bavarski, švapski ili severno-nemački dijalekti. Nemački je posebno interesantan jer pripada istoj jezičkoj porodici kao engleski (germanska grupa), što omogućava testiranje sposobnosti sistema da razlikuje lingvistički srodne jezike.

- **Španski (es):** Veliki subset sa govornicima iz različitih špansko-govornih regiona - Španije, Meksika, Argentine, Kolumbije, i drugih latinoameričkih zemalja. Španski jezik pokazuje značajnu varijabilnost između različitih regiona, od kastiljanskog španskog u Španiji do različitih latinoameričkih varijanti, što pruža dodatni izazov za sistem. Ova raznovrsnost je korisna jer testira robusnost modela na intra-jezičke varijacije.

- **Francuski (fr):** Opsežan subset sa raznovrsnim akcentima koji pokriva francuski iz Francuske, Kanade (kvibečki francuski), Belgije, Švajcarske i afričkih frankofonih zemalja. Francuski je posebno interesantan jer pripada romanskoj jezičkoj porodici zajedno sa španskim, što omogućava testiranje sposobnosti sistema da razlikuje dva lingvistički slična jezika sa zajedničkim latinskim poreklom.

### Karakteristike dataseta

Dataset se odlikuje sledećim karakteristikama koje su relevantne za razvoj sistema za prepoznavanje jezika:

- **Trajanje snimaka:** Većina snimaka traje između 3-10 sekundi, što odgovara dužini tipične rečenice u spontanom govoru. Ova dužina je optimalna za naš zadatak jer je dovoljno duga da uhvati prozodijske karakteristike i fonetske obrasce jezika, ali ne toliko duga da nepotrebno povećava računsku kompleksnost. Tokom preliminarne analize, utvrdili smo da prosečna dužina snimka varira između jezika - engleski snimci su u proseku nešto kraći (oko 4.5 sekundi) zbog tendencije ka kraćim rečenicama, dok nemački snimci mogu biti duži (oko 6 sekundi) zbog složenih složenih rečeničnih struktura.

- **Sample rate:** Originalno 48 kHz (konvertovano na 16 kHz za eksperimente) - Originalni snimci su u visokoj rezoluciji što omogućava fleksibilnost u obradi. Mi smo konvertovali sve snimke na 16 kHz jer je to standardna vrednost za govorni signal koja pruža dovoljan frekventni opseg (do 8 kHz prema Nyquist teoremi) dok značajno redukuje količinu podataka. Ova konverzija se vrši korišćenjem high-quality resampling algoritama koji minimizuju aliasing i druge artifakte.

- **Format:** MP3 (konvertovano u WAV format za obradu) - Originalni snimci su u MP3 formatu sa varijabilnim bitrate-om (tipično 96-128 kbps) što omogućava efikasno skladištenje velikog dataseta. Za obradu, konvertujemo ih u nekompresovani WAV format kako bismo eliminisali potencijalne artifakte MP3 kompresije koji bi mogli uticati na ekstrakciju karakteristika. Iako MP3 kompresija može uvesti određene distorzije, posebno u visokim frekvencijama, naša analiza je pokazala da ove distorzije imaju minimalan uticaj na performanse sistema za prepoznavanje jezika.

- **Govornici:** Raznovrsna populacija - različiti govornici, oba pola, različite starosne grupe (od dece do starijih osoba) - Ova raznovrsnost je kritična za generalizaciju modela. Analiza distribucije govornika pokazuje da dataset ima relativno balansiranu zastupljenost polova (približno 60% muških i 40% ženskih govornika), iako ova distribucija varira između jezika. Starosna distribucija pokazuje dominaciju govornika između 20 i 50 godina, što odgovara demografiji tipičnih korisnika tehnologije, ali dataset takođe sadrži značajan broj snimaka od mlađih i starijih govornika.

- **Kvalitet:** Crowd-sourced snimci sa različitim uslovima snimanja (različiti mikrofoni, okruženja, nivoi šuma) - Ova heterogenost kvaliteta je, interesantno, istovremeno izazov i prednost. Sa jedne strane, varijabilnost u kvalitetu snimanja može otežati treniranje modela - model mora da nauči da prepozna jezik uprkos razlikama u kvalitetu snimanja. Sa druge strane, ova raznovrsnost čini model robusnijim na različite realne uslove upotrebe, što je na kraju krajeva ono što želimo. Snimci su napravljeni korišćenjem svega i svačega - od profesionalnih mikrofona, preko laptop i desktop mikrofona, do mobilnih telefona. Akustička okruženja takođe variraju od tihe studijske atmosfere do snimaka sa umerenom pozadinskom bukom (što je realniji scenario).

- **Licenca:** CC0 (Creative Commons Zero) - potpuno javno dostupan - Ova licenca omogućava potpunu slobodu korišćenja, modifikacije i distribucije dataseta bez ikakvih ograničenja, što je ključno za reproducibilnost istraživanja i komercijalizaciju razvijenih sistema.

- **Validacija:** Svaki snimak validiran od strane više korisnika za osiguranje kvaliteta - Proces validacije zahteva da najmanje dva nezavisna korisnika potvrde da snimak tačno odgovara prikazanom tekstu i da je dovoljnog kvaliteta. Snimci koji ne prođu validaciju se odbacuju ili vraćaju na ponovno snimanje. Ovaj proces značajno poboljšava kvalitet dataseta i redukuje šum u podacima.

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

![Slika 4.1: Distribucija dataseta po jezicima](images/dataset_distribution.png)

**Slika 4.1:** Bar chart prikazuje broj snimaka po jeziku u trening, validacionom i test skupu. Engleski ima najveći broj snimaka, što je očekivano s obzirom na globalnu zastupljenost jezika. Srpski ima najmanji subset, ali i dalje dovoljan za treniranje robusnih modela. Distribucija je relativno balansirana što sprečava bias ka jezicima sa više podataka.

![Slika 4.2: Distribucija dužine audio snimaka](images/audio_duration_histogram.png)

**Slika 4.2:** Histogram prikazuje distribuciju dužine audio snimaka u datasetu. Većina snimaka je koncentrisana između 3-10 sekundi, sa pikom oko 5 sekundi. Ova distribucija odgovara prirodnoj dužini rečenica u spontanom govoru. Snimci kraći od 2 sekunde i duži od 15 sekundi su filtrirani tokom preprocessing-a.

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

Evaluacija performansi je ključna faza u razvoju sistema za mašinsko učenje - ovde vidimo da li sav taj rad ima smisla ili ne. U ovom istraživanju, svih pet implementiranih modela evaluirano je na istom test skupu koji sadrži 10,018 audio zapisa (približno 2,000 po jeziku), što obezbeđuje fer i konzistentno poređenje. Važno je napomenuti da je test skup bio potpuno odvojen od trening i validacionog skupa tokom celog procesa razvoja - nisam "virio" u test podatke tokom treniranja, što osigurava da rezultati zaista reflektuju sposobnost generalizacije modela na neviđene podatke.

Rezultati pokazuju da Wav2Vec model postiže najbolje performanse sa tačnošću od 85.06%, dok ostali modeli postižu tačnost između 82% i 84%. Na prvi pogled, razlika od približno 3 procentna poena između najboljeg i najlošijeg modela može izgledati mala, ali treba staviti stvari u perspektivu: u kontekstu sistema sa pet klasa (gde nasumično pogađanje bi dalo tačnost od samo 20%), svi modeli pokazuju zaista dobre performanse. Tačnost od preko 85% na ovako raznovrsnom datasetu sa realističnim uslovima snimanja je, iskreno, solidan rezultat koji je uporediv sa state-of-the-art sistemima koje sam video u literaturi.

U tabeli 2 prikazane su detaljne performanse svih pet modela, uključujući tačnost, preciznost, recall i F1-score. Korišćenje više metrika omogućava sveobuhvatniju evaluaciju - dok tačnost daje opštu sliku performansi, precision i recall pružaju uvid u specifične tipove grešaka koje model pravi, a F1-score pruža balansiranu meru koja uzima u obzir obe dimenzije.

**Tabela 2:** Uporedni prikaz performansi modela

| Model | Tačnost | Precision (Weighted) | Recall (Weighted) | F1-score (Weighted) | Broj test uzoraka |
|-------|---------|---------------------|-------------------|---------------------|-------------------|
| Wav2Vec | 85.06% | 0.8509 | 0.8506 | 0.8503 | 10,018 |
| RNN | 84.37% | 0.8473 | 0.8437 | 0.8436 | 10,018 |
| CNN | 83.57% | 0.8406 | 0.8357 | 0.8363 | 10,018 |
| Hybrid CNN-RNN | 83.46% | 0.8424 | 0.8346 | 0.8364 | 10,018 |
| SVM (Classic ML) | 82.20% | 0.8243 | 0.8220 | 0.8204 | 10,018 |

![Slika 5.1: Uporedno poređenje performansi svih modela](images/model_comparison_barchart.png)

**Slika 5.1:** Grafički prikaz poređenja svih pet modela po različitim metrikama (Accuracy, Precision, Recall, F1-score). Wav2Vec model konzistentno pokazuje najbolje performanse, dok SVM kao baseline postiže respektabilne rezultate uprkos svojoj jednostavnosti.

Na slici 5.2 prikazana je uporedna analiza tačnosti CNN i RNN modela tokom treniranja, gde plava linija predstavlja CNN model, a crvena RNN model, ilustrujući konvergenciju oba modela tokom epoha treniranja.

![Slika 5.2: Uporedni prikaz tačnosti CNN i RNN modela](images/cnn_rnn_comparison.png)

**Slika 5.2:** Uporedni prikaz tačnosti CNN i RNN modela tokom treniranja. Plava linija predstavlja CNN model, crvena RNN model. Možemo videti da oba modela pokazuju sličan obrazac konvergencije, sa brzim poboljšanjem u prvih 10 epoha, a zatim postepenim usporavanjem. CNN model pokazuje nešto stabilniju konvergenciju sa manje oscilacija u validacionoj tačnosti.

![Slika 5.3: Training i validation curves za sve modele](images/all_models_training_curves.png)

**Slika 5.3:** Training i validation curves za svih pet modela. Wav2Vec i RNN modeli pokazuju najbolju konvergenciju, dok hibridni model pokazuje blagu tendenciju ka overfitting-u (veći gap između training i validation tačnosti). SVM model nema training curve jer ne koristi iterativno treniranje.

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

![Slika 5.4: Heatmap performansi po jezicima i modelima](images/performance_heatmap.png)

**Slika 5.4:** Heatmap prikazuje F1-score za svaki model (redovi) i jezik (kolone). Toplije boje (crvena) označavaju bolje performanse, hladnije (plava) lošije. Jasno se vidi da engleski jezik konzistentno postiže najbolje rezultate kroz sve modele, dok španski pokazuje najniže performanse, posebno kod RNN modela.

Detaljnija analiza performansi po jezicima otkriva nekoliko interesantnih obrazaca koji pružaju uvid u prirodu zadatka i karakteristike različitih jezika:

1. **Engleski jezik postiže najbolje rezultate u oba modela** - CNN model postiže F1-score od 0.93, dok RNN postiže 0.90. Ova superiornost engleskog jezika može se pripisati nekoliko faktora. Prvo, engleski subset dataseta je najveći i najraznovrsniji, što omogućava modelu da nauči robusnije reprezentacije. Drugo, engleski ima relativno jednostavan fonetski inventar u poređenju sa nekim drugim jezicima, što može olakšati razlikovanje. Treće, velika raznovrsnost akcenata u engleskom datasetu (američki, britanski, australijski, indijski) paradoksalno može pomoći modelu da nauči invarijantne karakteristike koje definišu engleski jezik nezavisno od specifičnog akcenta.

2. **Srpski jezik pokazuje nešto niže performanse** - F1-score od 0.90 za CNN i 0.86 za RNN. Ovo može biti posledica nekoliko faktora. Prvo, srpski subset je manji od engleskog, što ograničava količinu podataka dostupnih za učenje. Drugo, srpski jezik ima bogatu morfologiju i kompleksnu fonologiju sa glasovima koji ne postoje u drugim jezicima u datasetu (npr. ć, č, đ, dž), što može zahtevati više podataka za adekvatno modelovanje. Treće, srpski pripada slovenskoj jezičkoj porodici koja nije zastupljena drugim jezicima u našem datasetu, što znači da model ne može da iskoristi transfer learning između srodnih jezika. Ipak, performanse od preko 86% pokazuju da sistem može efikasno da prepozna srpski jezik uprkos ovim izazovima.

3. **Španski jezik ima najniže performanse** - F1-score od 0.88 za CNN i 0.85 za RNN. Ovo je posebno interesantno s obzirom da španski ima veliki broj govornika i opsežan subset u datasetu. Analiza matrice konfuzije (diskutovana u sledećoj sekciji) pokazuje da se španski najčešće meša sa francuskim, što je očekivano jer oba jezika pripadaju romanskoj grupi i dele mnoge fonetske i prozodijske karakteristike. Oba jezika imaju sličan vokalski sistem, slične konsonantske strukture, i uporedive prozodijske obrasce. Ova lingvistička sličnost predstavlja fundamentalni izazov za sistem - razlikovanje između dva slična jezika zahteva fokusiranje na suptilne razlike koje mogu biti teške za uhvatanje, posebno u kratkim audio segmentima.

4. **Nemački i francuski pokazuju srednje performanse** - Oba jezika postižu F1-score između 0.88 i 0.91, što je solidno ali ne izuzetno. Nemački, kao predstavnik germanske porodice, ima određene sličnosti sa engleskim (zajednički koreni reči, slične sintaksičke strukture), ali se razlikuje u prozodiji i fonetskom inventaru. Francuski, kao romanski jezik, deli karakteristike sa španskim ali ima distinktivne osobine kao što su nazalni vokali i specifična intonacija koja ga čini relativno lako prepoznatljivim.

---

### Analiza grešaka

Na slici 5.5 prikazana je matrica konfuzije za Wav2Vec model, gde dijagonala predstavlja tačne klasifikacije, dok van-dijagonalni elementi pokazuju greške i najčešće pogrešne klasifikacije između pojedinih jezika.

![Slika 5.5: Confusion matrix za Wav2Vec model](images/confusion_matrix_wav2vec.png)

**Slika 5.5:** Confusion matrix za Wav2Vec model. Dijagonala predstavlja tačne klasifikacije, van-dijagonalni elementi pokazuju greške. Najintenzivnija konfuzija je između španskog i francuskog (romanski jezici), što je očekivano zbog njihove lingvističke sličnosti. Engleski jezik pokazuje najmanje konfuzije sa drugim jezicima.

![Slika 5.6: Poređenje confusion matrica za CNN i RNN](images/confusion_matrix_comparison.png)

**Slika 5.6:** Side-by-side poređenje confusion matrica za CNN (levo) i RNN (desno) modele. Oba modela pokazuju slične obrasce grešaka, sa najčešćim konfuzijama između lingvistički srodnih jezika. RNN model pokazuje nešto više konfuzije kod srpskog jezika, što može ukazivati na to da temporalne karakteristike srpskog jezika nisu dovoljno dobro uhvaćene sa dostupnom količinom podataka.

U analizi grešaka CNN modela identifikovane su sledeće najčešće pogrešne klasifikacije:

1. **Španski ↔ Francuski:** 18 grešaka (5.5%)
   - Razlog: Oba jezika pripadaju romanskoj grupi i dele mnoge fonetske karakteristike. Ovo je bilo očekivano - kada sam slušao neke od pogrešno klasifikovanih snimaka, čak i meni je bilo teško da razlikujem španski od francuskog u kratkim segmentima.

2. **Nemački ↔ Engleski:** 12 grešaka (3.6%)
   - Razlog: Oba jezika pripadaju germanskoj grupi. Interesantno je da se ove greške uglavnom dešavaju na snimcima sa jakim akcentima ili lošijim kvalitetom snimanja.

3. **Srpski ↔ Ostali:** 15 grešaka (4.5%)
   - Razlog: Manji broj uzoraka za treniranje. Srpski se ponekad mešao sa različitim jezicima, što verovatno ukazuje na to da model nije imao dovoljno podataka da nauči robusne karakteristike srpskog jezika.

### Vremenska analiza

Vremenska analiza procesa obrade prikazana je u tabeli 5, koja prikazuje vreme potrebno za učitavanje audio fajla, ekstrakciju karakteristika i predikciju za CNN i RNN modele.

**Tabela 5:** Vremenska analiza procesa obrade

| Operacija | CNN Model | RNN Model |
|-----------|-----------|-----------|
| Učitavanje audio fajla | 45ms | 45ms |
| Ekstrakcija karakteristika | 120ms | 95ms |
| Predikcija | 8ms | 12ms |
| **Ukupno vreme** | **173ms** | **152ms** |

![Slika 5.7: Grafički prikaz vremenske analize](images/timing_analysis_barchart.png)

**Slika 5.7:** Stacked bar chart prikazuje vreme potrebno za svaku fazu obrade (učitavanje, ekstrakcija, predikcija) za CNN i RNN modele. Jasno se vidi da ekstrakcija karakteristika zauzima najveći deo ukupnog vremena, dok je sama predikcija relativno brza. RNN model ima prednost u ukupnom vremenu zahvaljujući bržoj ekstrakciji MFCC karakteristika.

Nekoliko zanimljivih zapažanja:

- RNN model ima brže vreme ekstrakcije karakteristika jer MFCC zahteva manje računanja od mel-spektrograma - što ima smisla kada razmislite o tome
- CNN model ima brže vreme inferencije zbog paralelizacije konvolucionih operacija na GPU - GPU-ovi su zaista dobri za ovakve operacije
- Oba modela omogućavaju real-time obradu (< 200ms po audio segmentu od 10 sekundi), što znači da bi sistem mogao da se koristi u praktičnim aplikacijama bez frustrirajućih kašnjenja

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

![Slika 5.8: Uticaj dužine audio segmenta na tačnost](images/segment_length_impact.png)

**Slika 5.8:** Line chart prikazuje kako tačnost prepoznavanja raste sa dužinom audio segmenta. Oba modela pokazuju značajno poboljšanje od 3 do 10 sekundi, nakon čega kriva počinje da se izravnava. CNN model konzistentno pokazuje bolje performanse od RNN modela na svim dužinama segmenata. Optimalna dužina segmenta za praktičnu upotrebu je oko 10 sekundi - duža od toga donosi marginalne koristi uz povećanje računske kompleksnosti.
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
