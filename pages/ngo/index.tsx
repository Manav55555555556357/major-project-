// pages/dashboard.tsx
import React, { useState, useEffect } from 'react';
import styles from './index.module.css';
import Link from 'next/link';
import { auth, createDonationRequest, getRestaurantDonations, getRestaurants } from '../../utils/firebase';
import { RestaurantUser } from '../../interfaces';
import Loader from '../../components/Loader';

const Dashboard = () => {
    const [restaurants, setRestaurants] = useState<RestaurantUser[]>([]);
    const [selectedRestaurant, setSelectedRestaurant] = useState<RestaurantUser | null>(null);
    const [loading, setLoading] = useState<boolean>(true);
    const [requestingDonation, setRequestingDonation] = useState<boolean>(false);

    useEffect(() => {
        const fetchData = async () => {
            const restaurants = await getRestaurants();
            for (const restaurant of restaurants) {
                restaurant.donations = await getRestaurantDonations(restaurant.uid);
            }
            console.log(restaurants);
            setRestaurants(restaurants);
            setLoading(false);
        };
        fetchData();
    }, []);

    const handleCardClick = (restaurant: RestaurantUser) => {
        console.log(restaurant)
        setSelectedRestaurant(restaurant);
    };

    const handlePopupClose = () => {
        setSelectedRestaurant(null);
    };

    if (loading) {
        return <Loader />;
    }

    const handleRequestDonation = async (restaurant: RestaurantUser) => {
        // Replace these with the appropriate values
        setRequestingDonation(true);
        const ngoId = auth.currentUser.uid;
        const donationIds = restaurant.donations.map(donation => donation.id);
        await createDonationRequest(ngoId, donationIds);
        alert('Donation request sent!');
        setSelectedRestaurant(null);
        setRequestingDonation(false);
    };

    return (
        <div className={styles.container}>
            <div className={styles.header}>
                <h1 className={styles.title}>NGO Dashboard</h1>
                <Link href="/login">
                    <button className={styles.logoutButton}>
                        <span>Logout</span>
                    </button>
                </Link>
            </div>
            <div className={styles.cardsContainer}>
                {restaurants.map((restaurant) => (
                    <div key={restaurant.uid} className={styles.card} onClick={() => handleCardClick(restaurant)}>
                        <h2 className={styles.cardTitle}>{restaurant.userData.name}</h2>
                        <p className={styles.cardAddress}>{restaurant.userData.address}</p>
                    </div>
                ))}
            </div>
            {selectedRestaurant && (
                <div className={styles.popupContainer}>
                    <div className={styles.popup}>
                        <button className={styles.closeButton} onClick={handlePopupClose}>
                            &times;
                        </button>
                        <h2 className={styles.popupTitle}>{selectedRestaurant.userData.name}</h2>
                        <p className={styles.popupAddress}>{selectedRestaurant.userData.address}</p>
                        <h3 className={styles.popupSubtitle}>Donations:</h3>
                        {requestingDonation ? (
                            <Loader />
                        ) : selectedRestaurant.donations.length > 0 ? (
                            <>
                                <ul className={styles.popupDonations}>
                                    {selectedRestaurant.donations.map((donation) => (
                                        <li key={donation.id}>
                                            {donation.foodName} ({donation.quantity})
                                        </li>
                                    ))}
                                </ul>
                                <button
                                    className={styles.requestButton}
                                    onClick={() => handleRequestDonation(selectedRestaurant)}
                                >
                                    Request Donation
                                </button>
                            </>
                        ) : (
                            <p>No donations are available from this restaurant.</p>
                        )}
                    </div>
                </div>
            )}
            <style jsx global>
                {`
                    :root {
                        /* Color palette */
                        --primary-color: #61876E;
                        --secondary-color: #A6BB8D;
                        --tertiary-color: #EAE7B1;
                        --background-color: #3C6255;
                        --text-color: #FFFFFF;
                        --input-background-color: #A6BB8D;
                        --border-color: #EAE7B1;
                        --error-color: #E74C3C;
                    }
                    body {
                        background-color: #3C6255;
                        font-family: sans-serif;
                        margin: 0;
                        color: #EAE7B1;
                    }
                `}
            </style>


        </div>
    );
};

export default Dashboard;
