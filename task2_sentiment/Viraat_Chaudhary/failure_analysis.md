# Task 2 — Manual error analysis

Twenty selected error cases have populated annotation fields. The error types, explanations and proposed fixes were prepared with AI assistance; Viraat confirmed review of the Task 2 explanations and analysis before repository integration. The upload validator checked their structure and evidence fields; it did not establish unassisted authorship. Proposed causes and fixes remain hypotheses until tested on development data.

Twenty unique official test errors, five in each required group. The model was chosen using validation before test access.

## Error 1 — confident_false_positive

Model: experiment_1; official source ID: 29330. True=0, predicted=1, P(positive)=0.9999958276748657.

Slice memberships: short_reviews

Actual review:

> Wow love the place and everything is very clean and new!\n\nGreat place to come and relax worth a try!\n\nCheers,\n\nEric Van Nguyen\nVisited April 2012

Error type: Possible label-text mismatch

Explanation: The text is entirely positive: it praises the clean venue and recommends visiting. The prediction agrees with the visible text but conflicts with the recorded negative label. This is a possible label-text mismatch, not proof that the official label is wrong.

Testable fix: Independently audit similarly inconsistent training and validation examples against available rating metadata. Compare training with and without confirmed training-label issues on an unchanged validation set. Preserve the official test label and exclude this test row from future training.

## Error 2 — confident_false_positive

Model: experiment_1; official source ID: 12480. True=0, predicted=1, P(positive)=0.9999103546142578.

Slice memberships: short_reviews

Actual review:

> my husband had an omelette that was good. i had a blt, a little on the small side for $10, but bacon was great. Our server was awesome!

Error type: Mixed aspects with a value complaint

Explanation: The food and server receive praise, but the BLT is described as small for its price. The positive prediction may reflect stronger positive wording than the value complaint. The binary rating label does not reveal how the reviewer weighted these aspects.

Testable fix: Compare an aspect-aware sentence representation with global pooling, using training examples and a validation subset containing food, service and price contrasts. Check whether it improves validation macro-F1 on mixed-aspect reviews without changing this test result.

## Error 3 — confident_false_positive

Model: experiment_1; official source ID: 34454. True=0, predicted=1, P(positive)=0.9996898174285889.

Slice memberships: medium_reviews;contains_negation;high_oov_rate

Actual review:

> It's a really good concept but the execution in authentic flavors for each region falls a little short.  Every dish we tried was a bit off.  We had the Indian Butter Chicken, Thai Lemongrass Curry, Samosas (that was probably the best item) and some weird Korean Firecracker Riceball.  (I'm Korean and my Eomma would be like \""what the truck is this?\"")  Kind of reminds me of Trader Joe ethnic food but if you like the real, authentic deal then this place is probably not your speed.

Error type: Contrast and authenticity criticism

Explanation: The review starts with a good concept and names a best item, but criticizes execution, describes every dish as off, and says the restaurant may not suit people seeking authentic food. Its positive prediction misses the overall critical assessment. High-OOV membership alone does not prove unknown words caused the error.

Testable fix: On development data, compare retaining contrast words such as but, if and then with the existing stopword policy. Inspect training-vocabulary OOV tokens separately and evaluate authenticity/contrast examples using validation macro-F1.

## Error 4 — confident_false_positive

Model: experiment_1; official source ID: 20008. True=0, predicted=1, P(positive)=0.9993809461593628.

Slice memberships: medium_reviews;contains_negation

Actual review:

> Fun place but when you're in your early-mid 30s and have a tad bit of style other than local carny  then you will stick out like sore thumb.  We're really into vintage midcentury 1950s pinup style themed anything, but you really couldn't tell this is what the cannery is going for besides the picture on book of matches and the two bar names, \""pinups\"" and \""Marylyn's lounge\"" ....other than that the cover band rocks and I got rush week college wasted like I do every time in Vegas, or Whitney for that matter.

Error type: Mixed sentiment and audience suitability

Explanation: The reviewer calls the venue fun and praises the band, but criticizes the style and weak execution of its advertised theme. The prediction is positive despite the negative label. The text mixes enjoyment with dissatisfaction, so a single causal explanation cannot be established from the output alone.

Testable fix: Construct a training-derived mixed-sentiment subset with suitability and theme-execution complaints. Compare sentence-level aggregation against the current pooled representation on a fixed validation subset.

## Error 5 — confident_false_positive

Model: experiment_1; official source ID: 27375. True=0, predicted=1, P(positive)=0.9993663430213928.

Slice memberships: medium_reviews;contains_negation

Actual review:

> We had dinner here for four. Had an artichoke appetizer with pita chips, club sandwich and shared a huge sundae for dessert. The service was awesome; space impressive. But, I have to say that when a gem from NYC gets transported to Vegas, and gets \""blown up\"" into Vegas standards with all the glam, size, \""in your face\"", and over-the-top stuff that makes Vegas what it is, that NYC gem looses something and just isn't the same. \n\nWe were looking to be awed by this chocolate place and weren't. Maybe we should have ordered some chocolate pieces?

Error type: Unmet expectations despite positive aspects

Explanation: Praise for service, space and dessert is followed by disappointment that the venue was not awe-inspiring and lost the original location's character. The very confident positive prediction is inconsistent with the recorded negative label and the closing disappointment.

Testable fix: Evaluate a sentence-attention or sentence-order-aware pooling variant on training and validation reviews whose positive aspects are followed by unmet expectations. Compare validation macro-F1 and calibration with the current head.

## Error 6 — confident_false_negative

Model: experiment_1; official source ID: 36001. True=1, predicted=0, P(positive)=8.888084994396195e-05.

Slice memberships: short_reviews

Actual review:

> Good food, decent beer, terrible bar service. Girl with short blond hair behind bar was awful. I'm sure this places gets better with time. I'd come back!

Error type: Mixed aspects and return intention

Explanation: The food and beer receive positive comments, while bar service is called terrible and awful. The reviewer expects improvement and says they would return. The negative prediction may emphasize the strong service criticism over the return intention and positive aspects.

Testable fix: Use training-derived mixed-aspect examples to compare an auxiliary return-intention feature or objective with the current model. Evaluate on a fixed validation subset containing explicit return intentions and conflicting aspect sentiment.

## Error 7 — confident_false_negative

Model: experiment_1; official source ID: 22807. True=1, predicted=0, P(positive)=9.915272676153108e-05.

Slice memberships: medium_reviews;contains_negation

Actual review:

> EDIT: They really did change the service up since I last posted this.\n\nHorrible service.\n\nUsed to be my favorite pizza in the city (at a reasonable price), but I'm rethinking that. We just had an altercation with a server who refused to split a check when we were paying with cash. He then proceeded to disrespect the party at the table, telling us to 'not give him attitude about it.'\n\nSorry Bella Notte, but we're not children. I don't care if you're working hard - it doesn't give you any excuse to disrespect your paying customers like that.

Error type: Updated opinion with historical complaint

Explanation: The opening edit says the service changed, while most of the remaining text describes an earlier dispute and poor service. The negative prediction follows the historical complaint but conflicts with the positive label. The short update does not explicitly quantify satisfaction, so some ambiguity remains.

Testable fix: Compare a sentence-aware model that retains update markers and distinguishes updated opinions from quoted prior complaints. Train only on development data and evaluate a fixed validation subset of edited reviews.

## Error 8 — confident_false_negative

Model: experiment_1; official source ID: 3223. True=1, predicted=0, P(positive)=0.0001015038214973174.

Slice memberships: short_reviews

Actual review:

> I've been here two times. The first time was really good, but the second was very boring flavored. I had some lemon chicken that was mediocre.  \n\nFor the price,  the last time turned me off, but will probably go back again.

Error type: Conflicting visits and willingness to return

Explanation: The first visit was good, the second food experience was mediocre, and the reviewer still expects to return. The negative prediction is understandable from the most recent criticism, but it conflicts with the positive label. This is mixed sentiment across visits.

Testable fix: Compare a development-trained model with separate representations for visit-specific opinions and return intention. Evaluate validation performance on reviews describing multiple visits without treating willingness to return as an automatic positive label.

## Error 9 — confident_false_negative

Model: experiment_1; official source ID: 30793. True=1, predicted=0, P(positive)=0.000191104321856983.

Slice memberships: medium_reviews;contains_negation

Actual review:

> This place is so much better since they changed owners.\n\nMy wife and I went when it was the old owners, it was terrible.  We waited forever and the food never came before we walked out.  People were served before us that walked in after and my wife actually got her soup before me and I sat and waited while they \""made more\"".\n\nIt was horrible.\n\nNow its much better.  The staff are very friendly, they treat their customers very well and I have nothing but positive things to now say about this place.  Its much better with the new owners.

Error type: Temporal reversal after ownership change

Explanation: The text contrasts a terrible experience under the old owners with much better service under new owners and an explicitly positive current assessment. The very confident negative prediction fails to reflect the stated temporal reversal.

Testable fix: Retain temporal markers such as since, old, now and new, then compare sentence-order-aware aggregation on a training/validation subset of reviews contrasting past and current experiences. Measure validation macro-F1 and error rate.

## Error 10 — confident_false_negative

Model: experiment_1; official source ID: 3485. True=1, predicted=0, P(positive)=0.0002287133247591555.

Slice memberships: medium_reviews;contains_negation

Actual review:

> Update: \nEmailed 2 weeks ago, did not get a reply or call. .\n\nDid receive a message from Chris H and asked me to resend my email which I did today.  Will update outcome.\n\n*** Final Update - 7/27/2014 ***\n\nAfter a week of exchanging emails with Chris H., he personally re-evaluated and has honored our warranty claim and they offered a in store credit since our product is no longer available. \n\nChris H was very diligent and followed through on researching our claim and kept us updated each step of the way.

Error type: Resolved complaint in a final update

Explanation: The early update reports no response, but the final update says the warranty claim was honored and praises diligent follow-through. The negative prediction conflicts with the positive resolution. Earlier complaint language is a plausible contributor, but this has not been verified with a model ablation.

Testable fix: Compare explicit update-section segmentation with unsegmented reviews on development data. Evaluate whether aggregating the final update with the earlier context improves validation performance on resolved-complaint reviews.

## Error 11 — near_threshold_error

Model: experiment_1; official source ID: 29664. True=0, predicted=1, P(positive)=0.5002233982086182.

Slice memberships: short_reviews;high_oov_rate

Actual review:

> I'm looking for comedians to start a comedy night in Glendale  if your interested email me cutty20s@yahoo.com

Error type: Little sentiment evidence

Explanation: The text recruits comedians and gives contact information rather than evaluating a business. The positive probability is 0.500223, almost exactly at the decision threshold, which is consistent with weak sentiment evidence. High-OOV membership is recorded but does not establish an OOV mechanism.

Testable fix: Measure uncertainty and validation coverage-risk curves on a training-derived subset of promotional or non-review text. Test a validation-calibrated abstention option as a future extension, while retaining the current binary test prediction and threshold.

## Error 12 — near_threshold_error

Model: experiment_1; official source ID: 15540. True=1, predicted=0, P(positive)=0.4997131824493408.

Slice memberships: medium_reviews;contains_negation

Actual review:

> First impressions are the most important.\n\nI am so glad I stopped by this location. The lady that runs the front counter is so freaking funny.\n\nI didn't know what to order when I got there. I saw Italian sausage and I just went for it. \nIt comes with marinara sauce, onions, and bell peppers. \n\nThe subs are above average they definitely have a better taste and the onions were so freaking good. \n\nThe location is very clean attached to a 7-11.

Error type: Positive review near the threshold

Explanation: The review praises the funny staff member, food, onions and cleanliness. The phrase did not know what to order expresses uncertainty about an order rather than negative service. The positive probability of 0.499713 places the incorrect negative prediction barely below the threshold.

Testable fix: Compare the current stopword policy with a version that retains clause structure on development data, and evaluate positive reviews containing incidental negation. Check validation discrimination and calibration rather than tuning a threshold to this test example.

## Error 13 — near_threshold_error

Model: experiment_1; official source ID: 18014. True=1, predicted=0, P(positive)=0.49965834617614746.

Slice memberships: medium_reviews;contains_negation

Actual review:

> Honestly the quality of food is about a 3 star but because I support the mom and pop shops and used to go here almost every day during my breaks from the nearby community college it deserves a 4. I've met the owners a couple times and the atmosphere in the afternoon is pretty quiet sometimes which is very nice when you're trying to just have a nice break. There was nothing that I didn't like here and the pricing is good as well, its just you can tell the food was once frozen.

Error type: Mixed rating and double negation

Explanation: Food quality is modest and the food was once frozen, but the reviewer gives four stars, praises atmosphere and pricing, and says there was nothing they did not like. The model predicts negative at P(positive)=0.499658. Mixed aspects and double negation are plausible challenges.

Testable fix: Create a development-only validation subset for double negation and mixed ratings. Compare retaining function words or using a negation-aware auxiliary objective with the current preprocessing and model.

## Error 14 — near_threshold_error

Model: experiment_1; official source ID: 21051. True=1, predicted=0, P(positive)=0.4996180534362793.

Slice memberships: long_reviews;contains_negation

Actual review:

> I usually don't write reviews but reading the ones on this site helped me decide where to go, so I'm paying it forward. \n\nI haven't been to a traditional salon in FOREVER but I wanted to get a haircut and I didn't think my Dominican salon could handle it.  The salon is located in the back of the barber shop so you have to walk through it to get to the chairs.  Didn't feel weird at all though, all the guys were super friendly.  I booked a consultation first to make sure I would feel comfortable putting my hair in their hands. There were two ladies there, I showed them what I was thinking and they walked me through what the style would actually look like on my head.  They were very welcoming, I felt like I had known them forever.  Not the same experience I've had at salons in the past where nobody smiles and you feel like you're interrupting their lunch.  In the end I felt comfortable enough to make an appointment and I'm going back in a few hours!  Only negative was that they tried to talk me into getting a weave, or pieces, idk.  It's just not my thing. \n\nOther than that it's a positive environment all around and I'm looking forward to seeing how my haircut comes out!

Error type: Positive evaluation with incidental negation

Explanation: The reviewer repeatedly describes welcoming staff, comfort and a positive environment. Several negative constructions refer to past salons or say the visit did not feel weird. A minor complaint about a weave is present. The prediction is negative with P(positive)=0.499618, very close to the threshold.

Testable fix: Compare clause-aware sentence aggregation on training/validation reviews with positive overall sentiment and incidental negation. Evaluate the negation slice and positive-review recall before any future independent test.

## Error 15 — near_threshold_error

Model: experiment_1; official source ID: 19979. True=0, predicted=1, P(positive)=0.5006575584411621.

Slice memberships: long_reviews;contains_negation

Actual review:

> My wife thought of a great idea to grab some carb legs for my birthday so she found Finz. Upon arriving we knew it was an old Vinnie's which at times could be sketch. Upon entering you could tell this was not a normal restaurant because of the seating set up, not really a separation between the bar and the dinner tables.\nWe were greeted as we entered and led to a table of our choice. Waitress was great and was pretty attentive. The food was good, definitely worth the prices which are decent. I ordered the Crab Legs my wife got the Steam Pot, My Step Father got the Peel & Eat Shrimp and my mother got the Shrimp and Scallop Pasta. All dishes were great except the Shrimp and Scallops were a little over cooked but the Pasta sauce was great. \nNow the bad, shortly after arriving and enjoying our Spicy Honey Shrimp App (Not SO Honey) there was an altercation between the Lady working the bar and a So Called Patron that apparently spends 8K a year there, ha ha.  The young lady decided to curse loudly at the man and hit him with her purse as he tried to defend himself and get very loud. There were kids there and this was around 7:30 which is normal for a Friday night dinner. After the families cleared out and the night went on the patrons who were regulars began to conversate about the incident with major profanity laced statements. As a first timer I could not believe their was a manager there to handle the incident and take care of the language as folks were enjoying dinner. Better management is needed. \n\nI agree with other folks the food is good but the regulars are not so welcoming. Unfortunately you cannot pick your customers but customers can pick their place to eat. I suggest if you are a family stick to weekdays maybe but FINZ is not a Grill more of an bar with good food. \n\nI will say that our waitress did apologize for the actions of the bar tender  but other than that nothing was done to cater to the guest who were trying to enjoy dinner.

Error type: Positive food but negative safety and management

Explanation: The food and waitress are praised, but the review describes a disruptive confrontation, profanity around children and inadequate management. The positive probability is only 0.500658. The negative label is consistent with the serious complaint about the dining environment despite positive food comments.

Testable fix: Compare sentence-level aspect aggregation that separates food quality from safety and management complaints. Train on development examples and evaluate mixed-aspect validation macro-F1, checking encoded length before attributing any failure to truncation.

## Error 16 — slice_specific_failure

Model: experiment_1; official source ID: 37451. True=0, predicted=1, P(positive)=0.9982247948646545.

Slice memberships: medium_reviews;contains_negation;high_oov_rate

Actual review:

> Decor and atmosphere really good. Actually a bit upscale for its Eastowne Mall location. Food was good, but not inexpensive. See photo. Ordered ribs and the walleye. Walleye  was tasty and prepared well. Was it worth $20. No, more in the area of $12-14. Service was avg.  Free wi-fi. Too early to tell how this place will fare with lower cost alternatives within earshot.

Error type: Price-value criticism in the high-OOV slice

Explanation: The food and decor are praised, but the reviewer questions whether the meal was worth its price and mentions cheaper alternatives. The model predicts positive with confidence 0.998225 despite the negative label. The selected slice is high OOV, but token-level evidence is needed to establish its contribution.

Testable fix: Inspect OOV tokens on training/validation price-value reviews, then compare a from-scratch subword vocabulary with the word vocabulary under a comparable parameter budget. Measure high-OOV validation macro-F1 and price-value error rates.

## Error 17 — slice_specific_failure

Model: experiment_1; official source ID: 30086. True=1, predicted=0, P(positive)=0.0003640086797531694.

Slice memberships: medium_reviews;contains_negation

Actual review:

> Went there yesterday and found that the place has just CLOSED.  Another victim of a good chef with a bad location and business planning.  \n\nHint for future restauranteurs:  Don't open a Scottsdale restaurant in the summer unless you have the cash to survive with no business till the following March, advertise heavily, and don't locate your business where customers coming from one direction have to make a U-turn to get there, particularly if there's no good signage.\n\nThe Stacy's location in Phoenix remains open.

Error type: Business closure with ambiguous evaluative target

Explanation: The review reports a closure and criticizes location and planning while describing the chef as good. Its positive label conflicts with much of the visible criticism, and the model predicts negative confidently. The label may reflect appreciation of the restaurant rather than its business prospects, but that interpretation is uncertain.

Testable fix: Independently audit development reviews about closures and identify whether criticism targets food, staff or business circumstances. Compare entity/aspect-aware representations on a fixed validation subset while preserving this official label.

## Error 18 — slice_specific_failure

Model: experiment_1; official source ID: 29494. True=1, predicted=0, P(positive)=0.0010162440594285727.

Slice memberships: long_reviews;contains_negation

Actual review:

> UPDATED. \n\nMy initial very frustrated and dramatic review read as follows:\n\nBililng practices are at best negligent and at worst fraudulent.  \n\nI started going to the studio per a groupon.  I enjoyed the experience so much that I purchased a discounted 20 pack of classes.  When I purchased the classes, my credit card was charged.  However, the purchase didn't show up in my online account so I couldn't schedule classes.  I called The Studio and they told me that they'd credited the classes to another student but they did remedy the situation. \n\nFast forward about 6 months and I am reviewing my credit card statement.  Lo and behold a mysterious charge from the Studio shows up for $130.  I have not even been back there in months and certainly have not authorized any transactions to this business.  I am now disputing the charge via my credit card. \n\nThe yoga is great but all these terrible billing practices make it not worth your time in the long run.\n\nUpdates:\n\nAfter reading this, the studio owner went out of her way to contact me to try and correct the error.  She contacted me several times over a month and offered to resolve the issue in minutes if I called her.  I couldn't manage to find time to call her, but she refused to give up and tracked down my information and resolved the situation in a way that goes above and beyond what can be expected from a business owner.  I can certainly say that although I was frustrated by the two billing mixups, this business went out of its way to accomodate and help me, even when I couldn't go out of my way to try and correct the error.  I am thus updating my review to 4 stars- not quite 5 due to the initial errors- but they certainly get 5 for the way they managed this.  \n\nAlso, as previously mentioned, the yoga is great.

Error type: Long review with a positive resolution

Explanation: An earlier section alleges billing problems, but a later update says the owner resolved the issue beyond expectations and explicitly updates the rating to four stars. The negative prediction misses that current resolution. Long-review membership does not establish that the positive update was truncated.

Testable fix: Check retained token length and the encoded tail first. In a future development-only ablation, compare update-aware hierarchical sentence pooling with the current model, and test a longer sequence limit only if actual development truncation is confirmed.

## Error 19 — slice_specific_failure

Model: experiment_1; official source ID: 21215. True=1, predicted=0, P(positive)=0.0006118340534158051.

Slice memberships: medium_reviews;contains_negation

Actual review:

> Last night several parents came in with over 15 children to celebrate their 9 year olds 4th grade graduation at 9:15 pm. The bartender expressed that it was not a place to have children running around as it is against the law and a liability issue if anything were to happen to them on their premise. The children were running in and out of the bar while the parents continued to drink upstairs claiming to watch them in the open park way. After their third round of drinks the bartender told them they were no longer welcomed due to the fact the kids were unsupervised as well as the other customers had cleared out expressing \""whether the bar was a childcare center\"". One father even started swearing at her telling her it was none of her business what their children were doing. I am offended in that they completely took advantage of her, were rude, not to mention had their children with them at a bar at 10:00pm in the evening. I will say the bartender was very cordial and had mind to do the right thing.

Error type: Sentiment directed at customers rather than staff

Explanation: The text criticizes rude parents and unsupervised children, then praises the bartender for acting cordially and doing the right thing. The negative prediction may conflate criticism of customers with the evaluation of the venue employee. The recorded positive label fits the closing praise.

Testable fix: Compare a model using development-trained entity/aspect tags to distinguish sentiment toward customers from sentiment toward staff. Evaluate a fixed validation subset with multiple people and conflicting sentiment targets.

## Error 20 — slice_specific_failure

Model: experiment_1; official source ID: 34919. True=1, predicted=0, P(positive)=0.0008559006964787841.

Slice memberships: short_reviews;contains_negation

Actual review:

> These burgers taste like they're made with high end Big Mac patties! Not a lot of variety, and really nothing all that special, but what they do, they do well.

Error type: Qualified praise after criticism

Explanation: The review criticizes limited variety and ordinary burgers, but concludes that the restaurant does what it offers well. The negative prediction does not match the positive label. The closing concession carries praise despite negative wording earlier in this short review.

Testable fix: Compare retaining concession markers and clause-level pooling on development reviews containing qualified praise. Evaluate the short-review and negation validation slices without modifying the current official-test results.

