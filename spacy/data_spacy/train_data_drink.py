TRAIN_DATA = [
    ("I love coffee.", {"entities": [(7, 13, "FAVORITE_DRINK")]}),
    ("My favorite drink is tea.", {"entities": [(21, 24, "FAVORITE_DRINK")]}),
    ("He enjoys drinking soda.", {"entities": [(19, 23, "FAVORITE_DRINK")]}),
    ("I like coffee.", {"entities": [(7, 13, "FAVORITE_DRINK")]}),
    ("I would like to drink coke.", {"entities": [(22, 26, "FAVORITE_DRINK")]}),
    ("cola", {"entities": [(0, 4, "FAVORITE_DRINK")]}),
    ("Just some water, please.", {"entities": [(10, 15, "FAVORITE_DRINK")]}),
    ("It's mix fruit juice.", {"entities": [(5, 21, "FAVORITE_DRINK")]}),
    ("I want some milk.", {"entities": [(12, 16, "FAVORITE_DRINK")]}),
    ("Do you have almond milk?", {"entities": [(12, 23, "FAVORITE_DRINK")]}),
    ("She prefers sparkling water.", {"entities": [(12, 28, "FAVORITE_DRINK")]}),
    ("Can I get a glass of wine?", {"entities": [(21, 26, "FAVORITE_DRINK")]}),
    ("They ordered fresh orange juice for breakfast.", {"entities": [(19, 31, "FAVORITE_DRINK")]}),
    ("Our special today is homemade lemonade.", {"entities": [(21, 39, "FAVORITE_DRINK")]}),
    ("I'm craving a cup of hot chocolate right now.", {"entities": [(21, 34, "FAVORITE_DRINK")]}),
    ("Is there any green tea left?", {"entities": [(13, 22, "FAVORITE_DRINK")]}),
    ("He's a fan of craft beer.", {"entities": [(14, 25, "FAVORITE_DRINK")]}),
    ("I'll have the herbal tea, please.", {"entities": [(14, 25, "FAVORITE_DRINK")]}),
    ("She can't start her day without a latte.", {"entities": [(34, 40, "FAVORITE_DRINK")]}),
    ("Could you pass the peach iced tea?", {"entities": [(19, 34, "FAVORITE_DRINK")]}),
    ("I enjoy a smoothie after my workout.", {"entities": [(10, 18, "FAVORITE_DRINK")]}),
    ("He took a sip of his espresso.", {"entities": [(21, 29, "FAVORITE_DRINK")]}),
    ("We shared a bottle of sparkling cider.", {"entities": [(22, 38, "FAVORITE_DRINK")]}),
    ("I ordered a pina colada at the beach.", {"entities": [(12, 23, "FAVORITE_DRINK")]}),
    ("We all enjoyed the margaritas.", {"entities": [(19, 29, "FAVORITE_DRINK")]}),
    ("My brother loves root beer.", {"entities": [(17, 27, "FAVORITE_DRINK")]}),
    ("Can we have two mojitos, please?", {"entities": [(16, 24, "FAVORITE_DRINK")]}),
    ("She sipped on her chai latte while reading.", {"entities": [(18, 28, "FAVORITE_DRINK")]}),
    ("They're famous for their iced Americano.", {"entities": [(25, 40, "FAVORITE_DRINK")]}),
    ("Do you serve kombucha here?", {"entities": [(13, 21, "FAVORITE_DRINK")]}),
    ("A glass of merlot, please.", {"entities": [(11, 18, "FAVORITE_DRINK")]}),
    ("He ordered a flight of craft beers to taste.", {"entities": [(23, 34, "FAVORITE_DRINK")]}),
    ("Would you recommend a good scotch?", {"entities": [(27, 34, "FAVORITE_DRINK")]}),
    ("I'm looking for a non-alcoholic beer.", {"entities": [(18, 37, "FAVORITE_DRINK")]}),
    ("Our signature drink is the blue lagoon.", {"entities": [(27, 39, "FAVORITE_DRINK")]}),
    ("I'll try the raspberry lemonade.", {"entities": [(13, 32, "FAVORITE_DRINK")]}),
    ("She enjoys a good gin and tonic in the evening.", {"entities": [(18, 31, "FAVORITE_DRINK")]}),
    ("Can I have a virgin mojito?", {"entities": [(13, 27, "FAVORITE_DRINK")]}),
    ("He loves the taste of a well-brewed matcha.", {"entities": [(28, 35, "FAVORITE_DRINK")]}),
    ("I prefer a bold cabernet over other wines.", {"entities": [(11, 24, "FAVORITE_DRINK")]}),
    ("Would you like some elderflower cordial?", {"entities": [(20, 40, "FAVORITE_DRINK")]}),
    ("They serve a fantastic boba tea here.", {"entities": [(23, 31, "FAVORITE_DRINK")]}),
    ("I enjoy the occasional whiskey sour.", {"entities": [(23, 30, "FAVORITE_DRINK")]}),
    ("A pitcher of sangria for the table, please.", {"entities": [(13, 20, "FAVORITE_DRINK")]}),
    ("She ordered a diet coke with her meal.", {"entities": [(14, 23, "FAVORITE_DRINK")]}),
    ("Try our new summer special, peach lemonade!", {"entities": [(28, 43, "FAVORITE_DRINK")]}),
    ("I'd like a hot apple cider on this chilly day.", {"entities": [(11, 26, "FAVORITE_DRINK")]}),
    ("For a refreshing drink, opt for a cucumber cooler.", {"entities": [(34, 50, "FAVORITE_DRINK")]}),
    ("Pass me the bottle of Pinot Noir, please.", {"entities": [(22, 33, "FAVORITE_DRINK")]}),
    ("I'll have a double espresso to start my day.", {"entities": [(12, 27, "FAVORITE_DRINK")]}),
    ("She can never say no to a glass of Chardonnay.", {"entities": [(35, 46, "FAVORITE_DRINK")]}),
    ("For me, a Bloody Mary is the perfect brunch cocktail.", {"entities": [(10, 21, "FAVORITE_DRINK")]}),
    ("He's obsessed with that new pumpkin spice latte.", {"entities": [(28, 48, "FAVORITE_DRINK")]}),
    ("A refreshing mint julep is all I need on a hot day.", {"entities": [(13, 23, "FAVORITE_DRINK")]}),
    ("Her all-time favorite is a classic Margarita.", {"entities": [(35, 45, "FAVORITE_DRINK")]}),
    ("Do you have any special recommendations for IPA beers?", {"entities": [(44, 54, "FAVORITE_DRINK")]}),
    ("A cold lemon iced tea would be great right now.", {"entities": [(2, 21, "FAVORITE_DRINK")]}),
    ("I'm thinking of trying the elderberry wine tonight.", {"entities": [(27, 42, "FAVORITE_DRINK")]}),
    ("Could we get a pitcher of Sangria for the table?", {"entities": [(26, 33, "FAVORITE_DRINK")]}),
    ("We should definitely try the house special, a Mai Tai.", {"entities": [(46, 54, "FAVORITE_DRINK")]}),
    ("I heard their cold brew coffee is the best in town.", {"entities": [(14, 30, "FAVORITE_DRINK")]}),
    ("She would like to order a vanilla bean frappuccino.", {"entities": [(26, 51, "FAVORITE_DRINK")]}),
    ("An Arnold Palmer would perfectly quench my thirst.", {"entities": [(3, 16, "FAVORITE_DRINK")]}),
    ("He prefers his gin fizz with a splash of lime.", {"entities": [(15, 46, "FAVORITE_DRINK")]}),
    ("They brought out a bottle of vintage port for us.", {"entities": [(29, 41, "FAVORITE_DRINK")]}),
    ("Can I start with a sparkling prosecco, please?", {"entities": [(19, 38, "FAVORITE_DRINK")]}),
    ("I'm in the mood for a warm mulled cider tonight.", {"entities": [(22, 39, "FAVORITE_DRINK")]}),
    ("Let's toast with a flute of champagne!", {"entities": [(28, 38, "FAVORITE_DRINK")]}),
    ("I could really go for a robust malbec right about now.", {"entities": [(31, 37, "FAVORITE_DRINK")]}),
    ("I'll have an orange juice, freshly squeezed, please.", {"entities": [(13, 26, "FAVORITE_DRINK")]}),
    ("She ordered a large grapefruit juice for breakfast.", {"entities": [(20, 36, "FAVORITE_DRINK")]}),
    ("Do you have any pomegranate juice?", {"entities": [(16, 33, "FAVORITE_DRINK")]}),
    ("He prefers pineapple juice over soda.", {"entities": [(11, 26, "FAVORITE_DRINK")]}),
    ("Can I get a cranberry juice with ice?", {"entities": [(12, 27, "FAVORITE_DRINK")]}),
    ("Their mango smoothie is the best.", {"entities": [(6, 20, "FAVORITE_DRINK")]}),
    ("We'll share a pitcher of lemonade.", {"entities": [(25, 34, "FAVORITE_DRINK")]}),
    ("I love the mint julep here.", {"entities": [(11, 21, "FAVORITE_DRINK")]}),
    ("A watermelon shake would be great right now.", {"entities": [(2, 18, "FAVORITE_DRINK")]}),
    ("He's having a black coffee and a strawberry shake.", {"entities": [(14, 50, "FAVORITE_DRINK")]}),
    ("He's having a black coffee.", {"entities": [(14, 27, "FAVORITE_DRINK")]}),
    ("I would like to have a strawberry shake.", {"entities": [(23, 40, "FAVORITE_DRINK")]}),
    ("She enjoys sipping on a lychee martini.", {"entities": [(24, 39, "FAVORITE_DRINK")]}),
    ("I'll start with a kiwi juice.", {"entities": [(18, 29, "FAVORITE_DRINK")]}),
    ("Would you like a guava smoothie?", {"entities": [(17, 32, "FAVORITE_DRINK")]}),
    ("A glass of cherry soda for me.", {"entities": [(11, 22, "FAVORITE_DRINK")]}),
    ("He's crazy about avocado shakes.", {"entities": [(17, 32, "FAVORITE_DRINK")]}),
    ("Try the new blueberry mocktail!", {"entities": [(12, 31, "FAVORITE_DRINK")]}),
    ("I recommend the peach iced tea.", {"entities": [(16, 31, "FAVORITE_DRINK")]}),
    ("Can we get a round of apple spritzers?", {"entities": [(22, 38, "FAVORITE_DRINK")]}),
    ("I'm in the mood for a papaya blend.", {"entities": [(22, 35, "FAVORITE_DRINK")]}),
    ("They make an excellent carrot and ginger juice.", {"entities": [(23, 47, "FAVORITE_DRINK")]}),
    ("Our special is a raspberry lime fizz.", {"entities": [(17, 37, "FAVORITE_DRINK")]}),
    ("I'd like a cucumber cooler to beat the heat.", {"entities": [(11, 26, "FAVORITE_DRINK")]}),
    ("For dessert, I'll try the banana milkshake.", {"entities": [(26, 43, "FAVORITE_DRINK")]}),
    ("He ordered the summer berry punch.", {"entities": [(15, 34, "FAVORITE_DRINK")]}),
    ("A coconut water would be refreshing.", {"entities": [(2, 15, "FAVORITE_DRINK")]}),
    ("May I have a turmeric latte?", {"entities": [(13, 28, "FAVORITE_DRINK")]}),
    ("She loves the green detox smoothie in the mornings.", {"entities": [(14, 34, "FAVORITE_DRINK")]}),
    ("Let's order a round of passion fruit mojitos.", {"entities": [(31, 45, "FAVORITE_DRINK")]}),
    ("I'll have the elderflower spritz, please.", {"entities": [(14, 33, "FAVORITE_DRINK")]}),
    ("A ginger beer would be perfect now.", {"entities": [(2, 13, "FAVORITE_DRINK")]}),
    ("I can't wait to try the fig and honey milkshake.", {"entities": [(24, 48, "FAVORITE_DRINK")]}),
    ("Order me a tangerine fizz, it sounds refreshing.", {"entities": [(11, 26, "FAVORITE_DRINK")]}),
    ("For a hot day, a frozen berry slushie works wonders.", {"entities": [(17, 37, "FAVORITE_DRINK")]}),
    ("I've heard their apricot sunrise is to die for.", {"entities": [(17, 32, "FAVORITE_DRINK")]}),
    ("She'll have the tropical mango passion smoothie.", {"entities": [(16, 48, "FAVORITE_DRINK")]}),
    ("Can I have a sip of your cucumber mint agua fresca?", {"entities": [(25, 51, "FAVORITE_DRINK")]}),
    ("Let's split a pomegranate and peach sangria.", {"entities": [(12, 43, "FAVORITE_DRINK")]}),
    ("I prefer a classic lime margarita on the rocks.", {"entities": [(19, 47, "FAVORITE_DRINK")]}),
    ("Is the kiwi and strawberry lemonade good here?", {"entities": [(7, 35, "FAVORITE_DRINK")]}),
    ("I'll just have a plain old vanilla milkshake.", {"entities": [(27, 45, "FAVORITE_DRINK")]}),
    ("He's obsessed with the new dragon fruit energy drink.", {"entities": [(27, 53, "FAVORITE_DRINK")]}),
    ("She wants a rose water and lychee cocktail.", {"entities": [(12, 43, "FAVORITE_DRINK")]}),
    ("Would love some cocktail.", {"entities": [(16, 25, "FAVORITE_DRINK")]}),
    ("I want veggie soup.", {"entities": [(7, 19, "FAVORITE_DRINK")]}),
    ("Our table would like two pitchers of mint juleps, please.", {"entities": [(37, 49, "FAVORITE_DRINK")]}),
    ("I think a warm apple cider would hit the spot.", {"entities": [(10, 26, "FAVORITE_DRINK")]}),
    ("A non-alcoholic ginger beer suits me fine.", {"entities": [(2, 27, "FAVORITE_DRINK")]}),
    ("I'm craving something like a blackberry boba tea.", {"entities": [(29, 49, "FAVORITE_DRINK")]}),
    ("Do they make a good spiced rum punch here?", {"entities": [(20, 36, "FAVORITE_DRINK")]}),
    ("I'd love a glass of your finest oolong tea.", {"entities": [(32, 43, "FAVORITE_DRINK")]}),
    ("She's all about that avocado and kale smoothie.", {"entities": [(21, 47, "FAVORITE_DRINK")]}),
    ("Have you tried the watermelon and feta mocktail?", {"entities": [(19, 48, "FAVORITE_DRINK")]}),
    ("A hibiscus iced tea would be lovely, thank you.", {"entities": [(2, 19, "FAVORITE_DRINK")]}),
    ("I'm looking for something like a cherry cola float.", {"entities": [(33, 51, "FAVORITE_DRINK")]}),
    ("Do you serve a classic Arnold Palmer here?", {"entities": [(15, 29, "FAVORITE_DRINK")]}),
    ("I'd like the house special, a lavender prosecco spritz.", {"entities": [(30, 55, "FAVORITE_DRINK")]}),
    ("What's the recipe for that blue curacao lemonade?", {"entities": [(27, 49, "FAVORITE_DRINK")]}),
    ("Can I have the seasonal plum and thyme prosecco?", {"entities": [(24, 48, "FAVORITE_DRINK")]}),
    ("I'm in the mood for a matcha green tea latte.", {"entities": [(22, 45, "FAVORITE_DRINK")]}),
    ("Do you have elderflower soda in stock?", {"entities": [(12, 28, "FAVORITE_DRINK")]}),
    ("A chilled beetroot and ginger shot sounds energizing.", {"entities": [(2, 34, "FAVORITE_DRINK")]}),
    ("He ordered a double espresso with a twist of lemon.", {"entities": [(13, 51, "FAVORITE_DRINK")]}),
    ("cocktail", {"entities": [(0, 8, "FAVORITE_DRINK")]}),
    ("I want a mocktail", {"entities": [(9, 17, "FAVORITE_DRINK")]}),
    ("sex in the beach", {"entities": [(0, 16, "FAVORITE_DRINK")]}),
]