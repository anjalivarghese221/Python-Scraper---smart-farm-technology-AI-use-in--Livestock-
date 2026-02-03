"""
Create Sample Training Dataset for Sentiment Analysis
This script generates a representative training dataset with labeled sentiments
focused on smart farming and agricultural technology
"""

import json
import random


def create_training_dataset():
    """Generate labeled training data for sentiment analysis"""
    
    # Positive sentiment examples
    positive_examples = [
        "This AI dairy monitoring system has significantly improved our herd health management",
        "The smart sensors are amazing and have reduced labor costs by 30 percent",
        "Precision agriculture technology is revolutionizing how we farm",
        "Love the automated feeding system, cows are healthier and productivity is up",
        "These IoT devices are incredibly helpful for tracking livestock behavior",
        "The drone technology is fantastic for monitoring large pastures efficiently",
        "Smart farming apps make data tracking so much easier and more accurate",
        "Excellent results with the robotic milking system, highly recommend",
        "The AI predictions for crop yields have been remarkably accurate",
        "This technology is a game changer for small and medium farms",
        "Very impressed with how the sensors detect early signs of disease",
        "The automation has freed up so much time for other farm tasks",
        "Best investment we made was in precision agriculture equipment",
        "The smart irrigation system saves water and increases crop quality",
        "Really happy with the performance of these agricultural robots",
        "This technology makes farming more sustainable and profitable",
        "The data analytics platform provides valuable insights for decision making",
        "Great improvement in milk quality since implementing smart monitoring",
        "These innovations are helping us compete in modern agriculture",
        "The machine learning algorithms accurately predict optimal harvest times",
        "Wonderful experience with automated livestock tracking systems",
        "This smart technology is perfect for managing large dairy operations",
        "Fantastic reduction in waste and improved efficiency across the board",
        "The computer vision system identifies problems before they become serious",
        "Smart farming is the future and these tools prove it",
        "Excellent ROI on our investment in agricultural technology",
        "These sensors provide real-time data that helps prevent losses",
        "Love how the technology integrates seamlessly with our existing systems",
        "The predictive maintenance features save us money on equipment repairs",
        "This innovation is making agriculture more accessible to new farmers",
        "Very satisfied with the accuracy of the AI health monitoring",
        "The automated systems work reliably even in harsh weather conditions",
        "Great support team and the technology delivers on its promises",
        "These tools have transformed our approach to farm management",
        "Impressive results in both productivity and animal welfare",
        "The smart technology pays for itself within the first year",
        "Excellent precision in detecting and addressing crop issues early",
        "This represents a major leap forward for modern agriculture",
        "Very pleased with how easy it is to use these advanced systems",
        "The technology helps us farm more responsibly and efficiently",
        "Amazing how much data we can collect and analyze now",
        "These innovations are crucial for feeding a growing population",
        "The automation allows us to scale operations without adding labor",
        "Really appreciate how this technology reduces environmental impact",
        "Smart farming tools have exceeded all our expectations",
        "The ROI is clear and the benefits continue to grow",
        "This technology makes farming less stressful and more rewarding",
        "Fantastic accuracy in monitoring individual animal health metrics",
        "The precision equipment reduces input costs while boosting yields",
        "Very impressed with the reliability and durability of these systems"
    ]
    
    # Negative sentiment examples
    negative_examples = [
        "The smart farming technology is too expensive and not worth the investment",
        "These sensors constantly malfunction and provide inaccurate data",
        "Terrible experience with the AI system, it crashes frequently",
        "The automation failed during critical times causing significant losses",
        "This technology is overhyped and doesn't deliver on its promises",
        "Very disappointed with the poor customer support and buggy software",
        "The learning curve is too steep and the system is not user friendly",
        "Complete waste of money, traditional methods work better",
        "The IoT devices lose connectivity constantly in rural areas",
        "This technology requires too much maintenance and technical expertise",
        "The data analytics are confusing and not actionable for farmers",
        "Horrible integration issues with our existing farm equipment",
        "The robotic system broke down and repairs are extremely expensive",
        "Not reliable enough for critical livestock monitoring applications",
        "The technology creates more problems than it solves on our farm",
        "Poor quality sensors that need constant replacement",
        "This AI system makes incorrect predictions that hurt productivity",
        "The subscription costs are unreasonable for small farms",
        "Very frustrated with the complexity of these smart farming tools",
        "The technology is designed for large operations not family farms",
        "Terrible ROI and we regret purchasing this equipment",
        "The drones are impractical in bad weather conditions",
        "This automation takes away jobs from farm workers",
        "The software updates constantly break existing functionality",
        "Not impressed with the accuracy of the monitoring systems",
        "These smart tools are too fragile for real farm environments",
        "The technology requires internet that we don't have reliable access to",
        "Complete disappointment, the system failed within months of installation",
        "The promises made by vendors don't match reality on the farm",
        "Too many false alarms from the AI detection systems",
        "The technology is not ready for prime time in agriculture",
        "Very unhappy with the performance and reliability issues",
        "This innovation is more marketing hype than practical solution",
        "The costs keep adding up with subscriptions and maintenance",
        "Poor training materials make it impossible to use effectively",
        "The technology is incompatible with our farming practices",
        "Negative impact on our budget with little to show for it",
        "These systems are unnecessarily complex for basic farming tasks",
        "The smart equipment is unreliable during peak season when we need it most",
        "Not worth the hassle of installation and constant troubleshooting",
        "The technology creates data privacy concerns on our farm",
        "Terrible user interface that makes simple tasks complicated",
        "This AI makes poor recommendations that ignore local conditions",
        "The equipment is poorly designed for actual farm use",
        "Very dissatisfied with the lack of local technical support",
        "The technology fails to account for variability in agriculture",
        "Not suitable for organic or traditional farming methods",
        "The automation removes the human element that's crucial in farming",
        "These sensors provide too much data and no useful insights",
        "Complete failure in delivering the promised benefits to our operation"
    ]
    
    # Neutral sentiment examples
    neutral_examples = [
        "The smart farming technology has both advantages and disadvantages",
        "We are still evaluating the effectiveness of these AI systems on our farm",
        "The sensors provide data but we need more time to see real results",
        "Some features work well while others need improvement",
        "The technology is interesting but not sure if it's right for our operation",
        "We've seen mixed results with the automated monitoring systems",
        "The cost benefit analysis is still unclear after six months of use",
        "These tools require significant investment and the payoff is uncertain",
        "The smart equipment works as advertised but nothing exceptional",
        "We're testing different precision agriculture technologies to compare",
        "The AI system provides information but doesn't replace experience",
        "Some farmers report success while others have had problems",
        "The technology is evolving and we're watching developments closely",
        "It's too early to determine if this investment will be worthwhile",
        "The automation handles routine tasks but requires human oversight",
        "We use some smart farming tools but also rely on traditional methods",
        "The sensors are accurate but the insights need better interpretation",
        "This technology may work better for some types of farms than others",
        "We're gradually incorporating smart systems into our operations",
        "The results are acceptable but not as transformative as expected",
        "Some aspects of the technology are useful others not so much",
        "We're learning to use these tools more effectively over time",
        "The smart farming approach requires careful planning and implementation",
        "The technology has potential but needs further development",
        "We see incremental improvements but no dramatic changes yet",
        "The system performs adequately for basic monitoring purposes",
        "Different farms will have different experiences with this technology",
        "We're exploring how to integrate smart tools with existing practices",
        "The data is helpful but requires analysis to be actionable",
        "This represents one option among many for modern farmers",
        "The technology works but the learning process takes time",
        "We have some concerns but also see some benefits",
        "The smart systems complement rather than replace human judgment",
        "Results vary depending on how the technology is implemented",
        "We're monitoring performance before making further investments",
        "The equipment functions as described in the specifications",
        "This technology is part of a larger farm management strategy",
        "We're still in the adoption phase and adjusting our approach",
        "The system provides baseline functionality without standout features",
        "Farmers should carefully evaluate their specific needs before purchasing",
        "The technology serves its purpose but isn't revolutionary",
        "We use these tools alongside conventional farming practices",
        "The smart systems require ongoing attention and optimization",
        "Our experience has been average compared to other farm technologies",
        "The effectiveness depends on proper setup and maintenance",
        "We're collecting data to make informed decisions about expansion",
        "The technology has a place in modern agriculture among other tools",
        "Some features are more useful than others for our operation",
        "We're taking a measured approach to adopting smart farming technology",
        "The system meets basic requirements but doesn't exceed expectations"
    ]
    
    # Create training dataset
    training_data = []
    
    # Add positive examples
    for text in positive_examples:
        training_data.append({
            'text': text,
            'sentiment': 'positive'
        })
    
    # Add negative examples
    for text in negative_examples:
        training_data.append({
            'text': text,
            'sentiment': 'negative'
        })
    
    # Add neutral examples
    for text in neutral_examples:
        training_data.append({
            'text': text,
            'sentiment': 'neutral'
        })
    
    # Shuffle the data
    random.seed(42)
    random.shuffle(training_data)
    
    return training_data


def main():
    """Generate and save training dataset"""
    print("=" * 60)
    print("CREATING SENTIMENT TRAINING DATASET")
    print("=" * 60)
    
    training_data = create_training_dataset()
    
    # Count sentiments
    sentiment_counts = {
        'positive': sum(1 for item in training_data if item['sentiment'] == 'positive'),
        'negative': sum(1 for item in training_data if item['sentiment'] == 'negative'),
        'neutral': sum(1 for item in training_data if item['sentiment'] == 'neutral')
    }
    
    print(f"\nGenerated {len(training_data)} training examples:")
    print(f"  Positive: {sentiment_counts['positive']}")
    print(f"  Negative: {sentiment_counts['negative']}")
    print(f"  Neutral: {sentiment_counts['neutral']}")
    
    # Save to JSON file
    output_file = 'sentiment_training_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nTraining dataset saved to: {output_file}")
    print("\nSample examples:")
    for i, item in enumerate(training_data[:3], 1):
        print(f"\n{i}. [{item['sentiment'].upper()}]")
        print(f"   {item['text'][:80]}...")
    
    print("\n" + "=" * 60)
    print("Dataset creation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
