## 1. Contexte et valeur clinique

Le service des urgences du Centre Hospitalier Saint-Aurélien connaît une surcharge constante, accentuée aux heures de pointe par des effectifs de triage insuffisants. L'accueil des patients y repose sur une évaluation initiale qui n'établit pas un diagnostic mais attribue un niveau de priorité, de l'urgence maximale à la prise en charge différée. Un allongement des délais à cette étape fait peser un risque propre : celui qu'un cas critique ne soit pas identifié assez tôt.

Le présent rapport rend compte d'un prototype d'agent conversationnel destiné à assister cette évaluation initiale. Il couvre l'ensemble de la chaîne réalisée, de la constitution d'un corpus médical bilingue anonymisé jusqu'à l'exposition du modèle spécialisé par une interface de programmation, en passant par son adaptation par fine-tuning supervisé puis par alignement sur préférences. Il s'agit d'une preuve de concept : son objet est d'établir la faisabilité technique du dispositif et d'en éprouver la valeur clinique, non de livrer un système exploitable en l'état.

Le positionnement retenu, celui d'une assistance soumise à la validation d'un soignant, ne relève pas de la précaution de principe. L'étude randomisée jointe à la mission, conduite sur cinquante médecins répartis en deux groupes, observe que le modèle de langage employé seul obtient sur des vignettes diagnostiques des scores de raisonnement supérieurs à ceux des deux groupes de praticiens, mais que la mise à disposition de ce même modèle auprès des médecins n'améliore pas significativement leur performance. La performance brute d'un modèle ne se convertit donc pas mécaniquement en valeur clinique : c'est la conception de son intégration au travail réel qui en décide. Ce constat oriente le prototype vers un rôle d'appui à la décision du soignant, et non de substitution.

Le prototype établit la faisabilité de la chaîne technique complète, depuis la préparation conforme des données jusqu'à un service d'inférence fonctionnel aux temps de réponse compatibles avec un usage interactif. Il n'établit pas la fiabilité clinique du modèle obtenu. L'évaluation conduite met au contraire en évidence des défaillances de nature à compromettre un usage réel, analysées en section 6, dont l'identification constitue l'un des apports de ce travail : elles déterminent les conditions à réunir avant toute mise en exploitation, énoncées en section 7.

## 2. Méthodologie données

### 2.1 Sources et constitution du corpus

Le corpus d'entraînement supervisé agrège trois sources publiques : MedQuAD, corpus anglophone de questions-réponses médicales, ainsi que FrenchMedMCQA et MediQA pour la partie francophone. Cette composition répond à l'exigence d'un
corpus bilingue, le service d'urgences visé recevant des patients dans les deux langues. Le jeu de préférences destiné à l'alignement provient d'une quatrième source distincte, UltraMedical-Preference, qui fournit des paires de réponses ordonnées et n'entre à aucun moment dans le corpus supervisé. Les licences d'utilisation de chacune de ces sources ont été vérifiées et sont documentées dans le dépôt, de même que leur origine.

Le corpus supervisé compte environ 5 000 paires instruction-réponse. Ce volume constitue une cible dimensionnée pour le POC et non un objectif à maximiser : la qualité et l'homogénéité de format ont été privilégiées sur le volume.

### 2.2 Normalisation et schéma de métadonnées

Les trois sources supervisées présentent des formats hétérogènes, aucune n'étant nativement structurée pour une tâche de triage. Leur normalisation a été confiée à un modèle de langage léger, Mistral Small, chargé de reformuler
les réponses brutes selon un format uniforme. Ce recours à un modèle utilitaire évite un travail de reformatage manuel incompatible avec le calendrier du projet, tout en produisant une régularité de structure que le fine-tuning
supervisé peut ensuite installer dans le modèle.

Le schéma de métadonnées associé retient les symptômes, les antécédents, les constantes, la source, le niveau d'urgence et le service concerné. Il remplit deux fonctions : fixer la structure de sortie attendue du modèle, et rendre les exemples filtrables et auditables après coup, chaque paire restant rattachée à la source dont elle procède.

### 2.3 Anonymisation et conformité au RGPD

Les données mobilisées relèvent du domaine médical et appellent à ce titre un traitement d'anonymisation documenté. Celui-ci s'appuie sur Presidio, associé à un modèle linguistique de reconnaissance d'entités, la stratégie de masquage
retenue substituant aux entités détectées un marqueur de remplacement.

Le paramétrage a fait l'objet de plusieurs arbitrages, l'application des réglages par défaut produisant un taux de faux positifs incompatible avec un corpus médical.

Les entités de type organisation ont été exclues du périmètre de détection. Le vocabulaire clinique, en particulier les noms de protocoles, d'échelles et d'institutions médicales, y déclenchait des détections en nombre, alors que ces
entités ne constituent pas des données identifiant un patient. Leur masquage dégradait le corpus sans bénéfice de protection.

Les entités de localisation ont été conservées au niveau du pays. Cette information présente une valeur clinique propre, notamment en médecine des voyages, où l'origine géographique d'un séjour oriente une partie du raisonnement diagnostique, et son niveau de granularité ne permet pas d'identifier une personne.

Le seuil de confiance appliqué aux détections a été fixé à 0,5. Ce réglage arbitre entre deux risques opposés, un seuil élevé laissant subsister des données identifiantes, un seuil bas dégradant le corpus par masquage excessif.

Le cas des éponymes médicaux a exigé un traitement spécifique. Un nombre important de désignations cliniques comportent un patronyme, sans qu'il désigne un patient. Une liste d'exceptions a été constituée, complétée par des règles de contexte examinant les caractères qui précèdent et suivent immédiatement l'occurrence, de manière à distinguer un patronyme employé comme désignation d'une pathologie d'un patronyme désignant une personne.

Un contrôle de qualité a été conduit après masquage. Il établit un résiduel de l'ordre de 0,9 % d'occurrences relevant des catégories éponyme et localisation dans le jeu de préférences. Ce résiduel est documenté comme une limite connue
du traitement, aucune itération supplémentaire n'ayant été engagée au regard du périmètre du POC. Un défaut distinct, portant sur des marqueurs de remplacement subsistant dans le corpus d'entraînement, a par ailleurs été identifié
tardivement et est analysé en section 6.

### 2.4 Partitionnement et versionnement

Le corpus supervisé est partitionné en trois jeux, entraînement, validation et test. Un jeu d'évaluation clinique est maintenu strictement à l'écart de cette partition et n'est mobilisé qu'après entraînement, afin qu'aucun des exemples
servant à l'appréciation du comportement clinique n'ait été vu par le modèle.

L'ensemble des jeux de données est versionné avec DVC, le stockage effectif étant assuré par un espace distant tandis que le dépôt Git ne conserve que les références. Ce dispositif remplit deux fonctions : il satisfait l'exigence de versionnement du livrable, et il conserve la trace de chaque transformation appliquée aux données, chaque état du corpus demeurant identifiable et restituable. Cette réversibilité constitue le support de l'auditabilité requise par le cadre réglementaire.

## 3. Méthodologie d'entraînement

Le modèle retenu est Qwen3-1.7B-Base. Sa taille permet un entraînement complet sur un seul GPU de milieu de gamme, condition nécessaire au cadre du POC, et sa version de base, non ajustée aux instructions, a été préférée délibérément :
c'est le fine-tuning supervisé qui doit installer le comportement attendu, suivi d'instruction et format de sortie de triage, plutôt qu'un alignement générique préexistant dont il faudrait ensuite corriger les effets.

### 3.1 Fine-tuning supervisé et adaptation à rang faible

Le fine-tuning supervisé exploite les paires instruction-réponse issues des trois sources du corpus bilingue. Il ne s'agit pas seulement d'exposer le modèle à du vocabulaire médical : l'objectif est de lui faire produire, à partir d'une description de symptômes, une sortie structurée comportant un niveau d'urgence, une explication et une orientation, format qu'un modèle de base ne produit pas spontanément.

L'entraînement recourt à l'adaptation à rang faible (LoRA). Les poids du modèle pré-entraîné restent figés ; l'apprentissage porte sur des matrices de rang réduit insérées auprès des couches ciblées, dont la contribution s'ajoute à celle des poids d'origine. Le nombre de paramètres effectivement entraînés est ainsi ramené à une fraction du total, ce qui réduit d'autant la mémoire requise et la taille de l'artefact produit, l'adaptateur seul étant conservé. La configuration retenue fixe un rang de 16 et un facteur alpha de 32, appliqués sur les quatre projections d’attention, les modules "q_proj", "k_proj", "v_proj" et "o_proj".

Deux ajustements ont été imposés par la mémoire disponible sur le GPU utilisé. L'activation du recalcul des activations (gradient checkpointing) échange du temps de calcul contre de la mémoire en ne conservant pas les activations intermédiaires, et la longueur de séquence a été plafonnée à 512 tokens. Sans ces deux réglages, l'entraînement dépasse la mémoire disponible.

L'entraînement s'est exécuté sur une machine virtuelle Google Cloud Platform équipée d'un GPU NVIDIA L4, pour une durée d'environ 26 minutes. Le recours à une instance payante plutôt qu'à un environnement gratuit constitue un choix
délibéré : il supprime le risque d'interruption de session en cours d'entraînement, principal facteur de perte de temps identifié en amont du projet.

### 3.2 Alignement par préférences

L'alignement est appliqué au modèle issu du fine-tuning supervisé, et non au modèle de base. Cet ordre est déterminant : l'optimisation par préférences ajuste un modèle à partir des réponses qu'il produit lui-même, et son signal perd sa pertinence si le modèle ajusté génère des sorties éloignées de celles sur lesquelles les préférences ont été définies. Partir du modèle supervisé maintient l'alignement dans cette distribution.

L'optimisation directe des préférences (DPO) a été retenue plutôt qu'un apprentissage par renforcement à partir de retours humains. Cette dernière famille de méthodes suppose d'entraîner au préalable un modèle de récompense,
puis d'optimiser le modèle de langage contre ce modèle au sein d'une boucle de renforcement, soit deux entraînements successifs et trois modèles à maintenir en mémoire. Le DPO reformule le problème comme une optimisation directe sur les
paires de préférences, et supprime le modèle de récompense intermédiaire. L'usage de LoRA apporte une simplification supplémentaire : le modèle de référence nécessaire au calcul de la perte s'obtient en désactivant l'adaptateur, ce qui évite d'en conserver une seconde copie en mémoire. Les méthodes plus récentes de la même famille ont été écartées, le périmètre du POC ne justifiant pas d'en explorer les variantes.

Le jeu de préférences provient du corpus UltraMedical-Preference, restreint aux paires dont l'écart de qualité est le plus marqué, soit 5 000 paires retenues. L'entraînement a duré environ 43 minutes sur la même infrastructure.

Il a été conduit sur une seule époque. Ce choix relève du périmètre du POC : la recherche d'un gain marginal par prolongation de l'entraînement ou par exploration d'hyperparamètres n'entre pas dans les objectifs du prototype, et
une exposition répétée à un jeu de préférences de cette taille exposerait au surapprentissage.

### 3.3 Traçabilité et reproductibilité

Les deux entraînements ont été suivis au moyen de Weights & Biases, consignant les courbes de perte et les indicateurs de préférence. Les checkpoints produits sont versionnés avec DVC et stockés sur un espace distant, au même titre que les jeux de données, de sorte que chaque version du modèle reste rattachée à la version exacte des données dont elle procède. Les hyperparamètres et la graine aléatoire sont fixés et consignés, condition de reproductibilité des deux exécutions.

## 4. Architecture de déploiement

L'entraînement produit un adaptateur LoRA distinct des poids du modèle de base, séparation qui a servi lors de l'alignement, l'adaptateur désactivé tenant lieu de modèle de référence. Cette organisation n'est plus utile au service. Les poids de l'adaptateur ont donc été fusionnés dans le modèle de base pour produire un artefact unique. La fusion supprime la surcouche d'application de l'adaptateur à chaque passe, évite d'avoir à gérer la cohérence entre deux fichiers versionnés séparément, et permet de servir le modèle comme n'importe quel modèle standard, sans dépendance au moteur d'adaptateurs.

Le service d'inférence repose sur vLLM. Son intérêt tient à sa gestion mémoire par pages, qui limite la fragmentation du cache d'attention, et à son regroupement continu des requêtes, qui maintient le débit lorsque plusieurs requêtes arrivent simultanément. Ces mécanismes restent peu sollicités à l'échelle du POC, où les requêtes sont émises séquentiellement par un seul utilisateur. Le choix se justifie moins par le gain immédiat que par la continuité avec la cible : le moteur retenu ici est celui qui supporterait une charge réelle, ce qui évite de reconstruire la chaîne d'inférence au moment du passage à l'échelle. Le serveur est lancé sur la machine virtuelle équipée du GPU, dans une session persistante qui le maintient actif indépendamment de la connexion.

L'application exposée aux clients est une API FastAPI. L'endpoint `/triage` reçoit la description des symptômes du patient, validée en entrée par un schéma Pydantic qui rejette les requêtes malformées avant toute génération. L'API construit alors la requête d'inférence, la transmet au serveur vLLM, et restitue la réponse au format JSON. Il faut noter que cette structuration est opérée par l'API à partir du texte produit par le modèle : la validation Pydantic s'applique aux données entrantes, non au contenu généré, dont la conformité n'est garantie par aucun mécanisme à ce stade.

L'application est conteneurisée. L'image ne contient que l'API et ses dépendances directes, le serveur d'inférence restant hébergé sur la machine hôte pour conserver l'accès au GPU. La conteneurisation vise ici la reproductibilité du déploiement plutôt que l'isolation : elle fige les versions des dépendances et rend le lancement identique d'un environnement à l'autre,
condition nécessaire à l'automatisation par le pipeline d'intégration continue décrite plus loin.

L'accès à la démonstration ne repose sur aucune règle de pare-feu ouverte. Aucun port n'est exposé publiquement et l'API n'est joignable qu'à travers un tunnel SSH établi depuis un poste autorisé. Ce dispositif supprime toute surface d'attaque exposée sur le réseau et convient au périmètre d'un POC dont les utilisateurs sont identifiés et peu nombreux. Il ne constitue pas pour autant un modèle d'accès transposable en production : il n'assure ni authentification par utilisateur, ni journalisation des appels par appelant, ni limitation de débit, trois exigences que devrait couvrir une passerelle applicative dans un déploiement réel.

La mission place la traçabilité de chaque interaction parmi les exigences du système, en vue des audits médicaux. L'API journalise à ce titre les requêtes reçues et les réponses produites. Cette journalisation a d'abord été mise en défaut par un défaut de configuration du conteneur : faute de volume monté, les enregistrements étaient écrits dans le système de fichiers éphémère du conteneur, invisibles depuis la machine hôte et perdus à chaque redémarrage.
Le répertoire de journalisation a été monté depuis l'hôte, et la persistance des enregistrements a été vérifiée après redémarrage du conteneur. Deux limites subsistent, assumées au périmètre du POC : les journaux ne font l'objet
d'aucune rotation, ni d'aucune sauvegarde hors de la machine virtuelle qui les héberge.

Le déploiement est automatisé par un pipeline d'intégration et de déploiement continus reposant sur GitHub Actions. Déclenché à chaque proposition de fusion vers la branche principale, il exécute la suite de tests automatisés, construit l'image du conteneur, et conditionne la fusion à la réussite de ces vérifications. Le pipeline garantit ainsi qu'aucune version ne parvient à la branche principale sans avoir été validée, condition de maintenabilité pour un système appelé à recevoir des versions successives du modèle.

La sécurité de la chaîne de dépendances a fait l'objet d'un suivi distinct, au moyen de l'analyse automatique des dépendances du dépôt. Une vulnérabilité de sévérité élevée affectant la bibliothèque `cryptography`, dépendance transitive du gestionnaire de versions de données et du composant d'anonymisation, a été corrigée par épinglage d'une version non affectée, la compatibilité étant vérifiée sur les deux composants concernés. Dix vulnérabilités affectant le moteur d'inférence ont par ailleurs été corrigées par montée de version, obtenue en restreignant la résolution des dépendances à la plateforme Linux, la
résolution portant simultanément sur Windows retenant une version antérieure au correctif.

Trois vulnérabilités subsistent, de sévérité modérée à faible. Aucune ne dispose d'un correctif compatible avec les contraintes de version imposées par le moteur d'inférence, dépendance centrale du projet, et leur correction
supposerait de renoncer à un composant conditionnant un livrable. Chacune a fait l'objet d'une analyse de sa condition d'exploitation, aucune n'étant atteignable dans le contexte d'usage retenu, à savoir un poste de travail et
une machine virtuelle dédiée, sans exposition réseau ni accès multi-utilisateur. Elles sont traitées comme des risques acceptés et documentés, ce statut devant être réexaminé à toute mise en production effective.

## 5. Résultats et métriques

### 5.1 Indicateurs d'entraînement

L'alignement a été précédé d'un run pilote sur une cinquantaine de paires, destiné à valider le pipeline d'entraînement avant engagement du run complet. Sa fonction était la vérification technique de la chaîne, non l'établissement
d'une référence de performance. Le run complet a ensuite porté sur les 5 000 paires du jeu de préférences, sur une époque.

| Métrique            | Run complet | Run pilote (50 exemples) |
|---------------------|-------------|--------------------------|
| train_loss          | 0,5484      | :                        |
| rewards/accuracies  | 0,75        | 0,60 à 0,67              |
| rewards/margins     | 0,8392      | 0,06 à 0,09              |
| rewards/chosen      | +0,5528     | :                        |
| rewards/rejected    | -0,2864     | :                        |
| mean_token_accuracy | 0,631       | :                        |

Le modèle ordonne correctement les paires de préférences dans trois cas sur quatre. Les scores attribués aux réponses préférées et rejetées se séparent nettement, le premier devenant positif et le second négatif, pour une marge moyenne de 0,84. Ces valeurs établissent que l'entraînement a convergé sur le signal de préférence attendu, sans divergence ni effondrement de la perte.

### 5.2 Comparaison qualitative sur vignettes cliniques

Cinq vignettes cliniques ont été soumises aux modèles entraînés successifs, dont deux construites pour cibler des confusions diagnostiques à risque.

| Vignette                               | SFT (sans adaptateur) | SFT+DPO                                | Verdict DPO        |
|----------------------------------------|-----------------------|----------------------------------------|--------------------|
| Douleur thoracique (cardiaque)         | Diagnostic correct    | Diagnostic erroné + antécédent inventé | Dégradation        |
| Fièvre pédiatrique                     | Traitement pertinent  | Diagnostic non fondé + examen inventé  | Dégradation sévère |
| Céphalées légères                      | Cause bénigne         | Cause bénigne                          | Neutre             |
| Faiblesse + trouble de la parole (AVC) | Diagnostic correct    | Diagnostic erroné (TVP)                | Dégradation        |
| Douleur abdominale (appendicite)       | Diagnostic erroné     | Diagnostic correct                     | Amélioration       |

L'alignement améliore la forme des réponses, leur structure et leur positionnement sur l'échelle d'urgence, mais son effet sur la justesse diagnostique n'est pas uniforme : une amélioration, un cas neutre et trois dégradations, dont une portant sur un diagnostic sans fondement dans l'énoncé. Ces observations, ainsi que celles issues des tests sur l'endpoint, sont analysées en section 6.

### 5.3 Latence et robustesse

La latence a été mesurée sur cinq requêtes successives adressées à l'endpoint de triage, dans les conditions de la démonstration. Les temps de réponse s'échelonnent de 1,17 à 2,28 secondes, pour une moyenne de l'ordre de 1,57
seconde. Cet ordre de grandeur est compatible avec un usage interactif en situation de triage, où le soignant saisit une description puis attend une proposition d'orientation. Aucun test de charge n'a été conduit : la mesure porte sur des requêtes séquentielles émises par un utilisateur unique et ne préjuge pas du comportement du service en accès concurrent.

Trois cas limites ont été soumis à l'endpoint pour en éprouver la robustesse. Une requête au format invalide est rejetée par la validation du schéma avant toute génération, avec un code d'erreur explicite et un message identifiant le champ en cause. Une entrée vide et une entrée de longueur excessive ne provoquent pas d'interruption du service, mais donnent lieu à une génération : le modèle produit une réponse dans les deux cas, dépourvue de cohérence pour la seconde. Aucune défaillance du serveur n'a été observée sur les trois cas.

La robustesse du service est donc établie au sens de la disponibilité, le service ne tombant sur aucune des entrées éprouvées, mais non au sens de la pertinence de la sortie : une entrée que le système devrait refuser de traiter produit néanmoins une réponse mise en forme comme une réponse valide.

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

## 7. Roadmap de passage à l'échelle

Les défaillances établies en section 6 déterminent les conditions d'une poursuite du projet au-delà du prototype. Les chantiers décrits ci-dessous en découlent directement et sont présentés dans leur ordre de dépendance : aucune montée en charge n'est envisageable avant que soit disponible un instrument capable d'en mesurer les effets cliniques.

### 7.1 Garde-fous

Le premier chantier conditionne tous les autres. Le **mode A**, erreur de diagnostic différentiel, ne se referme par aucun réglage technique : il relève d'un jugement clinique que le dispositif d'évaluation actuel n'est pas en mesure de porter. Une validation digne de ce nom suppose un corpus de vignettes élargi, une notation en aveugle par au moins deux médecins, une procédure de consensus en cas de désaccord et une mesure de l'accord inter-évaluateurs. Cette évaluation est l'instrument sans lequel aucune des évolutions suivantes n'est vérifiable. Elle ne dispense pas de la contrainte de positionnement retenue dès l'origine : le système assiste le triage, il ne le décide pas, et sa sortie reste soumise à la validation d'un soignant.

Le **mode B**, fabrication d'éléments cliniques, est le plus sévère et appelle un traitement propre. Il procède de ce qu'aucun mécanisme ne rattache la réponse produite au contenu de l'énoncé. Trois dispositions y répondent. La première consiste à faire expliciter au modèle les éléments de l'énoncé qui fondent son raisonnement, de sorte qu'un élément non attribuable devienne repérable à la lecture. La deuxième consiste à installer, par les données d'entraînement autant que par la consigne système, un comportement d'abstention : en l'absence d'antécédent ou de résultat d'examen fourni, le modèle doit signaler l'information manquante plutôt que la produire. La troisième consiste à interposer une vérification automatique confrontant les éléments cliniques cités à ceux présents en entrée, au moins pendant la phase initiale d'exploitation. Cette dernière disposition a un coût de latence qu'il faudra mesurer au regard des temps de réponse actuels. L'hypothèse d'un volume de données supervisées insuffisant reste par ailleurs à éprouver, mais elle n'a pas été établie et ne saurait tenir lieu de correctif.

Le **mode C**, dérive hors vocabulaire fermé, admet la solution la plus directe. Le décodage contraint, supporté par le moteur d'inférence retenu, restreint les tokens candidats à ceux compatibles avec le schéma de sortie et rend une valeur hors liste impossible à produire, plutôt que détectable après coup. La validation du schéma, aujourd'hui limitée aux données entrantes, serait étendue à la sortie en second rideau.

Le **mode D**, subsistance de marqueurs d'anonymisation, se corrige en amont du modèle. Un contrôle automatique, appliqué au corpus après masquage et avant constitution des jeux d'entraînement, suffit à intercepter les occurrences résiduelles. Ce contrôle constitue l'étape manquante identifiée en section 6 et s'intègre au pipeline de préparation des données comme condition de validation.

### 7.2 Projection industrielle

La phase suivante prévoit le recours à des modèles de plus grande capacité, de l'ordre de 32 milliards de paramètres et au-delà, adossés à un corpus étendu mobilisant l'intégralité des sources disponibles plutôt que l'échantillon de 5 000 paires retenu pour le prototype. Plusieurs familles récentes de modèles ouverts constituent des candidats à évaluer à ce titre comme Lllama 3.3 70B de Meta, Qwen3.5-35B-A3B-Base ou Qwen3-30B-A3B-Base d’Alibaba ou encore gemma-4-31B de Google Deepmind.

La portée attendue de cette montée en capacité doit être appréciée avec prudence. Elle peut raisonnablement améliorer la finesse du raisonnement diagnostique sur les cas ambigus, donc réduire la fréquence du mode A. Elle ne supprime pas le mode B : un modèle de plus grande capacité formule ses fabrications de manière plus assurée et plus vraisemblable, ce qui les rend plus difficiles à repérer par un lecteur humain. La montée en taille déplace donc le risque sans le traiter, et ne saurait précéder la mise en place des garde-fous décrits ci-dessus.

### 7.3 Conditions de mise en production

Le passage en exploitation devrait être subordonné à la vérification préalable des conditions suivantes, chacune donnant lieu à une réponse binaire :

- aucun élément clinique fabriqué détecté sur un corpus d'au moins cinquante vignettes, évalué en aveugle par deux cliniciens ;
- aucun faux négatif sur les pathologies à fenêtre thérapeutique critique du même corpus, et taux d'erreur de diagnostic différentiel inférieur au seuil fixé conjointement avec l'équipe médicale ;
- accord inter-évaluateurs mesuré et jugé suffisant sur ce corpus ;
- aucune valeur hors liste fermée produite sur l'ensemble des sorties évaluées ;
- aucun marqueur d'anonymisation résiduel détecté dans les jeux d'entraînement ;
- authentification par utilisateur, journalisation nominative et rotation des journaux en place ;
- comportement du service vérifié en accès concurrent sous charge représentative.