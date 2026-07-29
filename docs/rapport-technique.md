## 6. Analyse critique et limites

Les défaillances observées lors de l'évaluation comparative et des tests sur l'endpoint ne relèvent pas d'un mécanisme unique. Elles se répartissent en quatre modes distincts par leur origine et par la nature du risque qu'ils portent, deux d'ordre clinique et deux d'ordre technique. Ils sont présentés ici selon cette typologie plutôt que dans l'ordre chronologique de leur
découverte, avant d'être rapportés aux indicateurs suivis pendant l'alignement, qui n'en ont signalé aucun.

### Mode A, erreur de diagnostic différentiel sur cas ambigu

Ce premier mode regroupe les cas où le modèle retient une hypothèse diagnostique erronée alors que les éléments discriminants figuraient dans l'énoncé. Il se distingue du mode B décrit plus loin sur un point déterminant : aucune donnée clinique n'est inventée. Le modèle exploite mal une information disponible, il ne fabrique pas d'information absente.

Le premier cas porte sur une douleur thoracique aiguë accompagnée d'essoufflement et de sueurs froides. Le modèle SFT identifie correctement un infarctus du myocarde et classe le cas en urgence maximale. Après alignement DPO, le modèle écarte l'hypothèse cardiaque au profit d'une crise de panique, alors que l'association de ces trois signes constitue précisément le tableau qui impose d'éliminer une cause coronarienne en priorité.

Le troisième cas présente une faiblesse soudaine associée à un trouble de la parole. Le modèle SFT oriente vers une cause vasculaire ou neurologique centrale et propose les examens attendus, dont l'IRM. Le modèle aligné dévie vers une thrombose veineuse profonde, hypothèse sans lien causal avec la symptomatologie décrite, ce qui masque le diagnostic d'accident vasculaire cérébral.

Le quatrième cas, une douleur abdominale localisée en fosse iliaque droite, produit le résultat inverse. Le modèle SFT retient à tort une colique rénale, tandis que le modèle aligné identifie correctement une appendicite aiguë. L'alignement corrige donc ici une erreur du modèle précédent : son effet sur la justesse diagnostique n'est pas uniformément négatif, il est instable.

Les cas 1 et 3 partagent la caractéristique qui en détermine la gravité. L'infarctus du myocarde et l'accident vasculaire cérébral sont deux pathologies dont le pronostic dépend directement du délai de prise en charge. Une orientation initiale erronée n'y produit pas seulement un diagnostic faux, elle consomme la ressource qui conditionne le pronostic. C'est ce qui les distingue du quatrième cas, où une erreur initiale aurait été rattrapable dans un délai plus large.

### Mode B, fabrication d'éléments cliniques absents de l'énoncé

Ce deuxième mode regroupe les cas où le modèle produit des éléments cliniques qui ne figurent pas dans l'énoncé, antécédents ou résultats d'examens, afin d'étayer une conclusion diagnostique. Il constitue une aggravation du mode précédent : le modèle ne se contente plus d'exploiter incorrectement l'information disponible, il introduit dans le dossier une information fausse.

Dans le premier cas, celui de la douleur thoracique aiguë, le modèle aligné n'écarte pas seulement l'hypothèse cardiaque. Il produit un antécédent absent de l'énoncé, une crise de panique survenue dix ans auparavant, et s'en sert pour justifier sa conclusion. La donnée fabriquée vient donc soutenir une requalification à la baisse du niveau d'urgence.

Le deuxième cas concerne une fièvre chez un enfant de quatre ans. Le modèle fabrique deux éléments distincts, un signe clinique, le signe de Kaposi, et un résultat d'examen jamais réalisé, une sérologie anti-VIH positive, pour appuyer
un diagnostic sans fondement dans l'énoncé. Aucun des deux éléments n'était présent en entrée.

Le sixième cas confirme ce comportement en conditions de démonstration. Testé via l'endpoint de triage exposé par l'API, le modèle produit une explication mentionnant des signes cliniques incorrects, dont une association entre hématothorax et lésions cutanées dorsales. Le mode de défaillance n'est donc pas circonscrit à l'évaluation hors ligne, il se manifeste sur le prototype tel qu'il serait présenté à un utilisateur.

La gravité propre de ce mode tient à l'indiscernabilité de ce qu'il produit. Une erreur de diagnostic différentiel reste une hypothèse, qu'un soignant peut contester à la lecture de l'énoncé. Un antécédent ou un résultat de sérologie énoncés comme des faits n'ont, dans la sortie du modèle, aucune marque qui les distingue des données réellement transmises. Ils sont susceptibles d'être repris tels quels dans le parcours de soin, et d'y orienter des décisions ultérieures sur une base inexistante.

### Mode C, dérive hors vocabulaire fermé

Le troisième mode change de nature. Il n'affecte pas la justesse du contenu clinique mais la conformité de la sortie au format attendu, et relève donc de l'ingénierie plutôt que de la sécurité du patient.

Le sixième cas, déjà cité au titre du mode précédent, produit un niveau d'urgence formulé comme une urgence massive, valeur absente de la liste fermée définie par le schéma, qui n'admet que les niveaux maximale, modérée et différée. La même génération porte ainsi deux défaillances distinctes.

La difficulté tient au caractère silencieux de cet écart. La valeur produite est plausible et se lit comme un niveau d'urgence légitime, de sorte qu'un relecteur humain ne la signalera pas nécessairement. Un système consommant la sortie de façon structurée plutôt qu'en texte brut n'obtiendrait pour sa part aucune correspondance dans la liste attendue, sans qu'aucune erreur explicite ne soit levée. La validation appliquée aux données entrantes par le schéma Pydantic ne s'étend pas au contenu généré en sortie, qui n'est à ce stade contraint par aucun mécanisme.

### Mode D, fuite de marqueurs d'anonymisation

Le quatrième mode se distingue des trois précédents par son origine. Il ne provient ni du raisonnement du modèle ni de la conformité de sa sortie, mais du pipeline de préparation des données, dont un artefact subsiste jusque dans le texte généré.

Le septième cas, testé via le conteneur de l'API, produit une réponse contenant deux marqueurs d'anonymisation non résolus, restitués tels quels : `<LOCATION>` et `[PATIENT]`. Ces marqueurs sont les valeurs de remplacement appliquées par Presidio lors du masquage. Leur présence n'établit donc aucune fuite de donnée personnelle, le masquage ayant précisément joué son rôle, mais indique qu'un sous-ensemble des paires d'entraînement a été intégré au corpus sans vérification postérieure au masquage, et que le modèle a appris à reproduire ces marqueurs comme des éléments de langage ordinaires.

La portée du constat dépasse l'artefact lui-même. Il révèle une étape de contrôle absente entre l'anonymisation et la constitution du jeu d'entraînement, alors que la traçabilité de chaque transformation des données constitue une exigence du cadre réglementaire retenu pour ce projet.

### Portée et limites des métriques de préférence

Ces quatre modes appellent une question de méthode : les indicateurs suivis pendant l'alignement, une exactitude de préférence de 0,75 et une marge de 0,84, ne signalent aucune anomalie. L'écart tient à ce que ces indicateurs mesurent réellement. L'exactitude de préférence rend compte de la fréquence à laquelle le modèle ordonne correctement deux réponses issues du jeu de préférences, et la marge de l'écart de score entre elles. Toutes deux évaluent un classement relatif au sein d'une distribution donnée, non la fidélité d'une réponse aux éléments fournis en entrée.

Aucune des deux ne rattache la réponse au contenu de l'énoncé. La fabrication d'un antécédent ou d'un résultat d'examen n'y est donc pas pénalisée, et se trouve même susceptible d'être favorisée : une réponse produisant des éléments absents paraît plus complète et plus assurée qu'une réponse s'en tenant aux données disponibles. **L'angle mort n'est pas neutre, il est orienté**.

S'y ajoute un effet d'agrégation. Ces indicateurs sont des moyennes calculées sur l'ensemble du jeu de préférences, tandis que les défaillances relevées ici sont rares et sévères. Une moyenne ne fait pas apparaître ce type de queue de distribution. Les métriques de préférence restent nécessaires pour vérifier que l'alignement s'est déroulé correctement, mais elles ne constituent pas un instrument de sécurité clinique, lequel suppose une évaluation dédiée et distincte.

### Limites de l’évaluation

Ces constats doivent être rapportés aux limites du dispositif qui les a produits. Sept cas ont été examinés, cinq en évaluation hors ligne et deux via l'endpoint de démonstration, ce qui autorise à établir l'existence des modes décrits mais non à en estimer la fréquence. D'autres modes peuvent exister sans avoir été rencontrés.

La méthode elle-même est faiblement contrôlée. Les réponses ont été appréciées par un évaluateur unique, non clinicien, connaissant le modèle à l'origine de chaque sortie, sans notation en aveugle, sans second évaluateur et sans mesure
d'accord inter-évaluateurs. L'étude de référence citée en introduction, à titre de comparaison, fait noter chaque cas par deux médecins certifiés travaillant en aveugle, avec recherche de consensus en cas de désaccord.

Les cas ne constituent pas davantage un échantillon : deux vignettes ont été construites pour cibler des confusions diagnostiques déjà suspectées, ce qui oriente les résultats vers les défaillances recherchées. Les conditions de
génération diffèrent enfin d'un canal à l'autre. L'évaluation vaut donc comme signal qualitatif justifiant une validation clinique dédiée, non comme mesure du niveau de risque.